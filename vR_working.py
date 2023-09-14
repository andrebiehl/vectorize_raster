def vectorize_process(input_file_path):


    from osgeo import gdal, ogr, osr
    import numpy as np
    import os
    import gc
    from mpi4py import MPI
    import logging
    import glob
    import geopandas as gpd
    import pandas as pd
    import time
    import sys


    # Set input_file and output_dir based on current structure with outputs and uploads folders
    input_file = os.path.join('uploads', 'input_file_path.geotiff') # Change the file name accordingly
    output_dir = 'outputs'

    # Create output directory if not exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Set up logging to write to a file
    logging.basicConfig(filename=os.path.join(output_dir, 'vectorize.log'), level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("logger configured")


    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()


    SUB_BLOCKS = 5  # Global constant for number of sub-blocks
    WORKER_DELAY = 0.01  # Delay in seconds for worker process

    def receive_srs_wkt(source, tag, status):
        srs_wkt = comm.recv(source=source, tag=tag, status=status)
        return srs_wkt

    def extract_full_image(input_file):
        dataset = gdal.Open(input_file, gdal.GA_ReadOnly)
        if dataset is None:
            logger.error(f"Failed to open the dataset from file: {input_file}")
            return None, None

        cols = dataset.RasterXSize
        rows = dataset.RasterYSize

        # Capture the spatial reference
        srs = osr.SpatialReference()
        srs.ImportFromWkt(dataset.GetProjection())
        srs_wkt = srs.ExportToWkt()  # Convert SRS to WKT format

        # Capture the geotransform
        geotransform = dataset.GetGeoTransform()

        # Read the entire image into a numpy array
        image = dataset.ReadAsArray(0, 0, cols, rows)

        # Close the dataset
        dataset = None

        return image, srs_wkt, geotransform


    full_image, srs_wkt, geotransform = extract_full_image(input_file)
    if full_image is None:
        logger.error(f"Failed to extract the image from file: {input_file}")
    else:
        logger.info("image extracted")

    # Extracting the full image dimensions
    img_rows = full_image.shape[0]
    img_cols = full_image.shape[1]

    # Set the block size based on a ratio of the image dimensions
    block_rows = int(img_rows * 0.1) # 10% of the total rows
    block_cols = int(img_cols * 0.1) # 10% of the total columns

    # Ensure that block size is at least 1
    block_rows = max(1, block_rows)
    block_cols = max(1, block_cols)

    blocks = []
    for row in range(0, img_rows, block_rows):
        for col in range(0, img_cols, block_cols):
            start_row = row
            start_col = col
            num_rows = min(block_rows, img_rows - start_row)
            num_cols = min(block_cols, img_cols - start_col)
            block = full_image[start_row:start_row + num_rows, start_col:start_col + num_cols]
            blocks.append((block, start_col, start_row))  # Save the x and y offsets too


    # gc.collect()
    logger.info("blocks created")


    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    tag_work = 1
    tag_end = 0

    def work_on_block(block, rank, srs_wkt, xoffset, yoffset, geotransform):
        # Convert the received WKT string back to an SRS object
        srs = osr.SpatialReference()
        srs.ImportFromWkt(srs_wkt)

        mem_drv = gdal.GetDriverByName('MEM')
        dest = mem_drv.Create('', block.shape[1], block.shape[0], 1, gdal.GDT_Byte)
        dest.GetRasterBand(1).WriteArray(block)


        # Adjust the GeoTransform to match the block's location
        block_geotransform = list(geotransform)  # Make a copy of the original geotransform
        block_geotransform[0] = geotransform[0] + xoffset * geotransform[1]  # Adjust the top left x
        block_geotransform[3] = geotransform[3] + yoffset * geotransform[5]  # Adjust the top left y

        dest.SetGeoTransform(block_geotransform)  # Set the GeoTransform to the new one

        out_shp = os.path.join(output_dir, f"output_{rank}_{xoffset}_{yoffset}.shp")
        out_driver = ogr.GetDriverByName("ESRI Shapefile")
        out_ds = out_driver.CreateDataSource(out_shp)
        out_layer = out_ds.CreateLayer("polygonized", srs=srs)
        pixelValueField = ogr.FieldDefn("pixelValue", ogr.OFTInteger)
        out_layer.CreateField(pixelValueField)

        try:
            logger.info(f'Block data shape: {block.shape}, Block data content: {block}')  # Log block data shape and content

            gdal.FPolygonize(dest.GetRasterBand(1), dest.GetRasterBand(1), out_layer, 0, [], callback=None)
        except Exception as e:
            logger.error(f'Error during polygonize: {e}')
            return f'Error during polygonize: {e}'  # Return error message
        else:
            if out_layer.GetFeatureCount() > 0:
                logger.info('Shapefile created successfully.')
            else:
                logger.warning('Shapefile created but no features detected.')

        out_ds.Destroy()

        # gc.collect()

        return 'Finished work and polygonized'


    def send_block(block, dest, tag):
        logger.info(f'Sending block to process {dest} with dimensions {block.shape} and tag {tag}')
        sub_blocks = np.array_split(block, SUB_BLOCKS, axis=0)  # Use constant instead of hardcoding
        comm.send(len(sub_blocks), dest=dest, tag=tag)
        for sub_block in sub_blocks:
            comm.send(sub_block, dest=dest, tag=tag)
        logger.info(f'Block sent to process {dest}')

    def receive_block(source, tag=1, termination_tag=0, status=None):
        logger.info(f'Process {rank} is about to receive block from process {source} with tag {tag}')

        # First receive the number of sub-blocks
        num_sub_blocks = comm.recv(source=source, tag=MPI.ANY_TAG, status=status)

        if num_sub_blocks is None:  # termination signal received
            logger.info(f'Slave {rank} received end tag')
            return None, status  # Return status as well to avoid breaking other parts of the code

        if status:
            logger.info(f'Process {rank} received number of sub-blocks {num_sub_blocks} from process {status.Get_source()} with tag {status.Get_tag()}')

        # Initialize an empty list to hold the sub-blocks
        sub_blocks = []

        # Receive each sub-block
        for _ in range(num_sub_blocks):
            sub_block = comm.recv(source=source, tag=tag, status=status)
            sub_blocks.append(sub_block)

        # Concatenate the sub-blocks to form the original block
        block = np.concatenate(sub_blocks)
        logger.info(f'Received block from process {source} with dimensions {block.shape}')
        return block, status

    def send_termination_tag(dest, tag=0):
        logger.info(f'Sending termination tag to process {dest}')
        comm.send(None, dest=dest, tag=tag)  # Use None as termination signal

    # Master process
    if rank == 0:
        logger.info('Master started')


        # Initial work distribution
        for i in range(1, min(12, comm.Get_size())):
            block, xoffset, yoffset = blocks[i-1]
            comm.send((block, xoffset, yoffset, srs_wkt, geotransform), dest=i, tag=tag_work)
            logger.info(f'Initial block sent to process {i}')

        logger.info('Initial work distribution done')

        # Counter for the current block
        current_block = min(12, comm.Get_size())

        # While there are blocks left to distribute
        while current_block < len(blocks):
            # Receive finished work from any slave
            status = MPI.Status()
            received = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)

            # If the received data is a finished work confirmation
            if isinstance(received, str) and received.startswith('Finished'):  # Adjusted to catch both successful and failed messages
                source = status.Get_source()
                logger.info(f'Slave {source} finished work')

                # Assign the next block to the slave that just finished
                block, xoffset, yoffset = blocks[current_block]
                comm.send((block, xoffset, yoffset, srs_wkt, geotransform), dest=source, tag=tag_work)
                current_block += 1
                time.sleep(WORKER_DELAY)  # Give worker some time


        logger.info('All work distributed')

        # Wait for all slaves to finish and receive finish confirmation
        for i in range(comm.Get_size() - 1): 
            status = MPI.Status()
            received = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
            if isinstance(received, str) and received == 'Finished work and polygonized':
                source = status.Get_source()
                logger.info(f'Slave {source} finished work')

        logger.info('All slaves finished')

        # send termination tag to all worker processes
        for i in range(1, comm.Get_size()):
            send_termination_tag(dest=i, tag=tag_end)


        # Get a list of all shapefiles
        shapefiles = glob.glob(os.path.join(output_dir, '*.shp'))


        # Initialize an empty list to hold dataframes
        dfs = []

        # Loop over all shapefiles and read them into GeoDataFrames
        for shapefile in shapefiles:
            df = gpd.read_file(shapefile)
            dfs.append(df)
            logger.info(f'Number of features in GeoDataFrame: {len(df)}')


        # Only concatenate if there are GeoDataFrames to concatenate
        if dfs:
            # Concatenate all GeoDataFrames
            combined = pd.concat(dfs, ignore_index=True)

            # Save to a new shapefile
            combined.to_file(os.path.join(output_dir, 'combined.shp', driver='GPKG', layer='name'))
            #, driver='GPKG', layer='name')   #GPKG has no limits for file size
            # Remove individual shapefiles and associated files
            for shapefile in shapefiles:
                os.remove(shapefile)  # Remove .shp file
                os.remove(shapefile[:-4] + '.dbf')  # Remove .dbf file
                os.remove(shapefile[:-4] + '.shx')  # Remove .shx file
                os.remove(shapefile[:-4] + '.prj')  # Remove .prj file
                os.remove(shapefile[:-4] + '.cpg')  # Remove .cpg file

                # If there are any other files related to the shapefile, remove them here
        else:
            logger.warning('No GeoDataFrames to concatenate.')

    else:
        while True:
            status = MPI.Status()
            logger.info(f'Slave {rank} waiting for data')
            data = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
            if data is None or status.Get_tag() == tag_end:
                logger.info(f'Slave {rank} received end tag')
                break
            block, xoffset, yoffset, srs_wkt, geotransform = data
            logger.info(f'Slave {rank} received block')
            result = work_on_block(block, rank, srs_wkt, xoffset, yoffset, geotransform)
            comm.send(result, dest=0)
            logger.info(f'Slave {rank} sent result')


return 'Vectorization process completed.'

