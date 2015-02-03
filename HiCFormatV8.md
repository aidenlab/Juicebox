|Field | Description |	Type | Value |
|------|------------|------|-------|
|Magic|HiC magic string|String|HIC|
|Version|Version number|int|8|
|mIdxPos|File position of master index|||
|genomeId|	Genome identifier (e.g. hg19, mm9, etc)|	String||	
|nAttributes	|Number of key-value pair attributes|	int||	
||*List of key-value pair attributes (n = nAttributes)*||
||key	|	String	||
||value|		String||	
|------|------------|------|-------|
|nChrs|	Number of chromosomes|int||		
||*List of chromosome lengths (n = nChrs)*||
|chrName	|Chromsome name	|String||	
|chrLength|	Chromosome length |	int	||
|------|------------|------|-------|
|nBpResolutions	|Number of base pair resolutions|	int||	
||*List of bin sizes for bp resolution levels (n = nBpResolutions)*||
|resBP	|Bin size in base pairs	|int||	
|------|------------|------|-------|
|nFragResolutions	|Number of fragment resolutions	|int||	
||*List of bin sizes for frag resolution levels (n = nFragResolutions)*||
|resFrag	|Bin size in fragment units (1, 2, 5, etc)|	int||	
|------|------------|------|-------|
||*List of fragment sites per chromosome, in same order as chromosome list above (n = nChrs).  This section absent if nFragResolutions == 0.)*||
|nSites|	Number of sites for this chromosome|	int||	
||*List of sites (n = nSites)*||
|site|	Site position in base pairs|	int||	
