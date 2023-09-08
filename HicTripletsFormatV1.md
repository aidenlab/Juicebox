# hic-triplets file format 

## Structure

* Header
* Body
    * 3D Tensor
    * 3D Block
* Footer
    * Master index
    
## Header

|Field | Description |	Type | Value |
|------|------------|------|-------|
|Magic|HICT magic string|String|HICT|
|Version|Version number|int|1|
|masterIndexPosition|File position of the master index section in the footer (footer not finsihed yet). |long||
|normVectorIndexPosition|File position of the normalization vectors index in the footer (footer not finsihed yet). |long||
|normVectorLengthPosition|File position of the normalization vectors length in the footer (footer not finsihed yet). |long||
|genomeId|	Genome identifier (e.g. hg19, mm9, etc)|	String||	
||||
|nAttributes	|Number of key-value pair attributes|	int||
||*List of key-value pair attributes (n = nAttributes).  See notes on common attributes below.*||
|key	|Attribute key|	String	||
|value|Attribute value|		String||	
|||||
|nChrs|	Number of chromosomes|int||		
||*List of chromosome lengths (n = nChrs)*||
|chrName	|Chromosome name	|String||	
|chrLength|	Chromosome length |	int	||
|||||
|nBpResolutions	|Number of base pair resolutions|	int||	
||*List of bin sizes for bp resolution levels (n = nBpResolutions)*||
|resBP	|Bin size in base pairs	|int||	

## Body

The **Header** section is followed immediatly by the **Body**, which containe the three-way contact map data for each 
chromosome-chromosome-chromosome triplet and each resolution.   

### 3D Tensor Metadata

This section contains metadata  for the 3D contact tensors.  It is repeated for all each chromosome-chromosome-chromosome triplet.  
The master index contains an entry for each combination and is used to randomly access a specific
3D tensor as needed.  The metadata in this section includes an index for data blocks which contain the actual
contact data.  


|Field	|Description|	Type|	Value|
|------|------------|------|-------|
|chr1Idx| Index for chromosome 1.  This is the index into the array of chromosomes defined in the header above.  The first chromosome has index **0**.|	int||	
|chr2Idx| Index for chromosome 2. |	int	||
|chr3Idx| Index for chromosome 3. |	int	||
|nResolutions	|Total number of resolutions for this chromosome-chromosome-chromosome triplet, including base pair and fragment resolutions.	|int||	
||*Resolution metadata.  Repeat for each resolution. (n = nResolutions)*||
|unit|	Distance unit, base-pairs or fragments	|String	|BP or FRAG|
|resIdx	|Index number for this resolution level, an Array index into the bin size list of the header (resBP), first element is **0**. |	int||	
|sumCounts|	Sum of all counts (or scores) across all bins at current resolution.|	float||	
|occupiedCellCount|	Total count of cells that are occupied.  **Not currently used**|float|0|		
|percent5|	Estimate of 5th percentile of counts among occupied bins. **Not currently used**|float|0|		
|percent95|	Estimate of 95th percentile of counts among occupied bins **Not currently used**|float|0|		
|binSize|	The bin size in base-pairs	|int||	
|blockBinCountX			|Dimension of each 3D block in bins in x direction.  3D intra Blocks are cubes, and 3D inter Blocks are roughly cubic.|int||
|blockBinCountY			|Dimension of each 3D block in bins in y direction.  |int||
|blockBinCountZ			|Dimension of each 3D block in bins in z direction.  |int||
|blockXCount|The count of blocks along the x-axis in the 3D grid of blocks.  |int||	
|blockYCount|The count of blocks along the y-axis in the 3D grid of blocks.  |int||	
|blockZCount|The count of blocks along the z-axis in the 3D grid of blocks.  |int||	
|blockCount|The number of blocks stored in the file.  Note empty blocks are NOT stored.|||			
|||||
|*Block index. Repeat for each resolution  (n = nResolutions) <br> Note the order is metadata_zoom1, block_index_zoom1, metadata_zoom2, block_index_zoom2,...<br>*||
|blockNumber	|Numeric id for block.  This is the linear position of the block in the grid when counted in a slice-major, then row-major order.   ```blockNumber = z * (blockXCount * blockYCount) + y * blockXCount + x``` where first row and column  **0**	  |int|	
|blockPosition|	File position of block|	long|	
|blockSizeBytes	|Size of block in bytes| int|	
||||	
||*Block data*||
| blocks | Compressed blocks for all 3D blocks and resolutions.  See  description below.   |||


#### 3D Block  

A block represents a roughly cubic 3D sub-block of a contact block. 

***Note: Blocks are indivdually compressed with ZLib***

|Field	|Description|	Type|	Value|
|------|------------|------|-------|
|nRecords	|Number or contact records in this block|	int	|
|binXOffset | X offset for the contact records in this block.  The binX value below is relative to this offset.||
|binYOffset | Y offset for the contact records in this block.  The binY value below is relative to this offset.
|binZOffset | Z offset for the contact records in this block.  The binZ value below is relative to this offset.
|useShort | Flag indicating whether the ```counts``` for contact records for this block are recorded with data type ```short```.  If == 1 a ```short``` is used, otherwise type is an ```float```| byte |
|useShortBinX | Flag indicating whether the ```x position``` of contacts for this block are recorded with data type ```short```.  If == 1 a ```short``` is used, otherwise type is an ```int```| byte |
|useShortBinY | Flag indicating whether the ```y position``` of contacts for this block are recorded with data type ```short```.  If == 1 a ```short``` is used, otherwise type is an ```int```| byte |
|useShortBinZ | Flag indicating whether the ```z position``` of contacts for this block are recorded with data type ```short```.  If == 1 a ```short``` is used, otherwise type is an ```int```| byte |
|blockRepresentation | Representation of matrix used for the contact records.  If == 1 the representation is a ```list of slices (list of rows)```, if == 2 ```dense (not implemented)```. | byte |
|blockData| The block matrix data.  See descriptions below.

##### Block data - list of list of rows

The outer list represents the list of 2D slices of the 3D block. The inner
list represents the list of rows within a particular 2D slice.

|Field	|Description|	Type|	Value|
|------|------------|------|-------|
|sliceCount | Number of nonempty 2D slices. The data type is determined by the ```useShortBinZ``` flag above. | short : int ||
||
|*slices (n = sliceCount)*
|sliceNumber | 2D slice number, first slice is ```0```. The data type is determined by the ```useShortBinZ``` flag above. | short : int ||
|rowCount | Number of nonempty rows for this 2D slice. Slice is sparse, empty rows are not recorded. The data type is determined by the ```useShortBinY``` flag above. | short : int ||
||
|*rows (n = rowCount)*
|rowNumber | row number in the 2D slice, first row is ```0```. The data type is determined by the ```useShortBinY``` flag above. | short : int ||
|recordCount | Number of records for this row. Row is sparse, zeroes are not recorded. The data type is determined by the ```useShortBinX``` flag above. | short : int ||
||
|*contact records (n = cellCount)*||
|binX	|X axis index. The data type is determined by the ```useShortBinX``` flag above.|	short : int||
|value	|Value (counts or score). The data type is determined by the ```useShort``` flag above.|	short : float||	

##### Block data - dense
Not implemented yet. The memory overhead will be huge.

### Footer

| Field |	Description|	Type |	Value |
|------|------------|------|-------|
|nBytesV5|	Number of bytes for the “version 5” footer, that is everything up to the normalized expected vectors.  This field (*nBytesV5*) is not included, so the total number of bytes between ```footerPosition``` and ```nNormVectors```  is ```nBytesV5 + 4```. |long||	

#### Master index

| Field |	Description|	Type |	Value |
|------|------------|------|-------|
|nEntries|	Number of index entries|	int||
||	
||*List of index entries (n = nEntries)*||
|key|	A key constructed from the indeces of the three chromosomes for this 3D tensor.  The indeces are defined by the list of chromosomes in the header section with the first chromosome occupying index **0**|String||	
|position	|Position of the start of the chromosome-chromosome-chromosome 3D tensor record in bytes	|long||	
|size	|Size of the chromosome-chromosome-chroosome 3D tensor record in bytes.  This does not include the **Block** data.| int||	

#### Expected value vectors

Not implemented yet.

#### Normalized expected value vectors

Not implemented yet.

#### Normalization vectors

Not implemented yet.