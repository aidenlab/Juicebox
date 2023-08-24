# hierc file format 

## Structure

* Header
* Body
    * 3D Tensor
    * 3D Block
* Footer
    * Master index
    * Expected value vectors



## Header

|Field | Description |	Type | Value |
|------|------------|------|-------|
|Magic|HierC magic string|String|HIERC|
|Version|Version number|int|1|
|footerPosition|File position of the Footer section, containing the master index, expected values, and normalization vectors (footer not finsihed yet). |long||
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
|||||
|nFragResolutions	|Number of fragment resolutions	|int||	
||*List of bin sizes for frag resolution levels (n = nFragResolutions)*||
|resFrag	|Bin size in fragment units (1, 2, 5, etc)|	int||	
|||||
||*List of fragment site positions per chromosome, in same order as chromosome list above (n = nChrs).  This section absent if nFragResolutions = 0.*||
|nSites|	Number of sites for this chromosome|	int||	
||*List of sites (n = nSites)*||
|sitePosition|	Site position in base pairs|	int||	

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
|resIdx	|Index number for this resolution level, an Array index into the bin size list of the header, first element is **0**. |	int||	
|sumCounts|	Sum of all counts (or scores) across all bins at current resolution.|	float||	
|occupiedCellCount|	Total count of cells that are occupied.  **Not currently used**|int|0|		
|percent5|	Estimate of 5th percentile of counts among occupied bins. **Not currently used**|float|0|		
|percent95|	Estimate of 95th percentile of counts among occupied bins **Not currently used**|float|0|		
|binSize|	The bin size in base-pairs or fragments	|int||	
|blockSize			|Dimension of each 3D block in bins.  3D Blocks are cubes, so the total number of bins is ```blockSize^3```.  See description of grid strcture below|int||
|blockColumnCount|The number of columns in the grid of blocks.  |int||			
|blockCount|The number of blocks stored in the file.  Note empty blocks are NOT stored.|||			
|||||
|*Block index. Repeat for each resolution  (n = nResolutions)*||
|blockNumber	|Numeric id for block.  This is the linear position of the block in the grid when counted in row-major order.   ```blockNumber = column * blockColumnCount + row``` where first row and column **0**	|int|	
|blockPosition|	File position of block|	long|	
|blockSizeBytes	|Size of block in bytes| int|	
||||	
||*Block data*||
| blocks | Compressed blocks for all 3D blocks and resolutions.  See  description below.   |||


#### 3D Block  

A block represents a cubic 3D sub-block of a contact block. 

***Note: Blocks are indivdually compressed with ZLib***

|Field	|Description|	Type|	Value|
|------|------------|------|-------|
|nRecords	|Number or contact records in this block|	int	|
|binXOffset | X offset for the contact records in this block.  The binX value below is relative to this offset.||
|binYOffset | Y offset for the contact records in this block.  The binY value below is relative to this offset.
|binZOffset | Z offset for the contact records in this block.  The binZ value below is relative to this offset.
|useFloat | Flag indicating the ```value``` field in contact records for this block are recorded with data type ```float```.  If == 1 a ```float``` is used, otherwise type is ```short```| byte |
|blockRepresentation | Representation of matrix used for the contact records.  If == 1 the representation is a ```list of list of rows```, if == 2 ```dense```. | byte |
|blockData| The block matrix data.  See descriptions below, also  in the notes section.

##### Block data - list of list of rows

The outer list represents the list of 2D slices of the 3D block. The inner
list represents the list of rows within a particular 2D slice.

|Field	|Description|	Type|	Value|
|------|------------|------|-------|
|sliceCount | Number of 2D slices | short ||
||
|*slices (n = sliceCount)*
|sliceNumber | 2D slice number, first slice is ```0``` | short ||
|sliceCount | Number of nonempty rows for this 2D slice. Slice is sparse, empty rows are not recorded. | short ||
||
|rowCount | Number or rows | short ||
||
|*rows (n = rowCount)*
|rowNumber | Matrix row number, first row is ```0``` | short ||
|recordCount | Number of records for this row. Row is sparse, zeroes are not recorded. | short ||
||
|*contact records (n = cellCount)*||
|binX	|X axis index|	short||
|value	|Value (counts or score). The data type is determined by the ```useFloat``` flag above.|	float : short||	

##### Block data - dense
|Field	|Description|	Type|	Value|
|------|------------|------|-------|
|nRecords | Number of contact records in this 3D block.  | int ||
|w | Width of the dense block.  This can be < the blockSize if the edge columns on either side are zeroes.  See discussion on block representation below | short ||
||
|*contact records (n = nRecords)*||
|value	|Value (counts or score). The data type is determined by the ```useFloat``` flag above.|	float : short||	

### Footer

| Field |	Description|	Type |	Value |
|------|------------|------|-------|
|nBytesV5|	Number of bytes for the “version 5” footer, that is everything up to the normalized expected vectors.  This field (*nBytesV5*) is not included, so the total number of bytes between ```footerPosition``` and ```nNormVectors```  is ```nBytesV5 + 4```. |int||	

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

| Field |	Description|	Type |	Value |
|------|------------|------|-------|
|nExpectedValueVectors|	Number of expected value vectors to follow.  These are expected values from the non-normalized observed matrix.| int|	
||
||*List of expected value vectors (n = nExpectedValueVectors)*||
|unit|	Bin units either FRAG or BP.	|String	|FRAG : BP|
|binSize	|Bin (grid) size for this calculation	|int||	
|nValues	|Size of the vector|	int||	
||
|*List of expected values (n = nValues)*|
|value	|Expected value|	double||	
|nChrScaleFactors| Number of chromosome normalization factors| int||
||
||*List of normalization factors (n = nChrScaleFactors)*||
|chrIndex|	Chromosome index|	int||	
|chrScaleFactor|	Chromosome scale factor	|double||	


#### Normalized expected value vectors
| Field |	Description|	Type |	Value |
|------|------------|------|-------|
|nNormExpectedValueVectors|	Number of normalized expected value vectors to follow	|int||	
||
|*List of normalized vectors (n = nNormExpectedValueVectors)*||
|type|	Indicates type of normalization	|String|	VC:KR:INTER_KR:INTER_VC:GW_KR:GW_VC|
|unit	|Bin units either FRAG or BP.	|String|	FRAG : BP|
|binSize|	Bin (grid) size for this calculation	|int||	
|nValues|	Size of the vector	|int	||
||
||*List of expected values (n = nValues)*||
|value	|Expected value	|double||	
||
|nChrScaleFactors|Number of normalizatoin factos for this vector|||
||*List of normalization factors (n = nChrScaleFactors)*||
|chrIndex|	Chromosome index	|int	||
|chrScaleFactor|	Chromosome scale factor	|double||	

#### Normalization vectors
| Field |	Description|	Type |	Value |
|------|------------|------|-------|
|nNormVectors|	Number of normalization vectors |	int||	
||*List of normalization vectors (n=  nNormalizationVectors)*||
|type	|Indicates type of normalization	|String|	VC:KR:INTER_KR:INTER_VC:GW_KR:GW_VC|
|chrIdx|	Chromosome index	|int|	|
|unit|	Bin units either FRAG or BP.|	String|	FRAG : BP|
|binSize	|Resolution 	|int||	
|position|	File position of value array|	long	||
|nBytes|	Size in bytes of value array	|int	||
||*Normalization vector arrays (repeat for each entry above)*||
|nValues|	Number of values in array|	int||	
||*Normalization vector values (n=  nValues)*||
