/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2011-2015 Broad Institute, Aiden Lab
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 *  THE SOFTWARE.
 */

package juicebox.tools;

import juicebox.HiC;
import juicebox.data.*;
import juicebox.windowui.HiCZoom;
import juicebox.windowui.NormalizationType;
import org.broad.igv.Globals;
import org.broad.igv.feature.Chromosome;

import java.io.IOException;
import java.util.List;

/**
 * Created by nchernia on 1/9/15.
 */
public class Arrowhead {

    public static void runArrowhead(String file, int resolution) throws IOException {

        System.err.println("This method is not currently implemented.");
        System.exit(1);


        // might need to catch OutofMemory errors.  10Kb => 8GB, 5Kb => 12GB in original script
        DatasetReaderV2 reader = new DatasetReaderV2(file);


        Dataset ds = reader.read();

        if (reader.getVersion() < 5) {
            throw new RuntimeException("This file is version " + reader.getVersion() +
                    ". Only versions 5 and greater are supported at this time.");
        }

        // Should possibly check for KR, if doesn't have it, exit.

        // For each chromosome, launches blockbuster_sub_list.sh res file chr
        // then concatenates them together.

    //   will need to deal with getting corner scores for lists, possibly different function?
        // First calls arrowhead with 1-2000, 1000-3000, etc. sized matrices.
        List<Chromosome> chromosomes = ds.getChromosomes();

        // Note: could make this more general if we wanted, to arrowhead calculation at any BP or FRAG resolution
        HiCZoom zoom = new HiCZoom(HiC.Unit.BP, resolution);

        for (Chromosome chr : chromosomes) {
            if (chr.getName().equals(Globals.CHR_ALL)) continue;

            Matrix matrix = ds.getMatrix(chr, chr);

            if (matrix == null) continue;
            MatrixZoomData zd = matrix.getZoomData(zoom);
            // M 1000 0.4
            int maxSize = (int)Math.ceil(chr.getLength() / (double)zoom.getBinSize());
            for (int i=0; i<maxSize; i+=1000) {
                List<Block> blockList = zd.getNormalizedBlocksOverlapping(i, i, i + 2000, i + 2000, NormalizationType.KR);
                // For block list, make a matrix and fill it out.  Not sure what to use yet though.  Not sparse, in any event.
                /*
                if (b != null) {
                    for (ContactRecord rec : b.getContactRecords()) {
                        float counts = rec.getCounts();
                        int x = rec.getBinX();
                        int y = rec.getBinY();
                        */
            }


        }
    }

/**
 * function [final, scores] = run_blockbuster(M,str,list, control)

 if (nargin < 3)
 list=[];
 control=[];
 end
 % get large number of blocks
 [result1, scores, control_scores]=call_blockbuster(M, 1000, 0.4, list, control);
 p_sgn=0.4;
 while (size(result1,1)==0 && p_sgn > 0)
 p_sgn = p_sgn-0.1;
 [result1, scores, control_scores]=call_blockbuster(M, 1000, p_sgn, list, control);
 end

 % high variance threshold, fewer blocks, high confidence
 result2 = call_blockbuster(M, 0.2, 0.5);

 resdiff = setdiff(result1,result2, 'rows');
 % only interested in big blocks.
 result1=resdiff(resdiff(:,2)-resdiff(:,1)>60, :);
 % result2 is in.  then take only bigger blocks in "block deserts"
 for i=1:size(result1,1)
 conflicts=false;
 for j=1:size(result2,1)
 % if the beginning of an existing block is within this interval,
 % or the end is within this interval, this is a conflicting block
 if ((result2(j,1) >= result1(i,1) && result2(j,1) <= result1(i,2)) || (result2(j,2) >= result1(i,1) && result2(j,2) <= result1(i,2)))
 conflicts=true;
 end
 end
 if (~conflicts)
 result2(end+1,:) = result1(i,:);
 end
 end

 result=result2;

 fprintf('\n');
 scores=merge_scores(scores);
 control_scores=merge_scores(control_scores);

 if (size(result,1)==0)
 final = [];
 return;
 end

 s = struct;
 s.minx=result(1,1); s.maxx=result(1,1);
 s.miny=result(1,2); s.maxy=result(1,2);
 s.value = result(1,:);
 C{1} = s;
 for i=2:size(result,1)
 added=false;

 for j=1:length(C)
 if ((abs(C{j}.minx - result(i,1)) < 5 || abs(C{j}.maxx - result(i,1)) < 5) && (abs(C{j}.miny - result(i,2)) < 5 || abs(C{j}.maxy - result(i,2)) < 5))
 % add to this bin
 added=true;
 C{j}.value(end+1,:) =result(i,:);
 if (result(i,1) < C{j}.minx); C{j}.minx = result(i,1); end
 if (result(i,1) > C{j}.maxx); C{j}.maxx = result(i,1); end
 if (result(i,2) < C{j}.miny); C{j}.miny = result(i,2); end
 if (result(i,2) > C{j}.maxy); C{j}.maxy = result(i,2); end
 break; % don't look anymore in C for bins
 end
 end
 if (added ~= true)
 s=struct;
 s.value=result(i,:);
 s.minx=result(i,1);
 s.maxx=result(i,1);
 s.miny=result(i,2);
 s.maxy=result(i,2);
 C{end+1}=s;
 end
 end
 final = zeros(length(C),4);
 for i=1:length(C)
 final(i,1)=C{i}.maxx;
 final(i,2)=C{i}.maxy;
 final(i,3)=mean(C{i}.value(:,3));
 final(i,4)=mean(C{i}.value(:,4));
 final(i,5)=mean(C{i}.value(:,5));
 final(i,6)=mean(C{i}.value(:,6));
 final(i,7)=mean(C{i}.value(:,7));
 end

 result=final;
 clear C;
 s = struct;
 s.minx=result(1,1); s.maxx=result(1,1);
 s.miny=result(1,2); s.maxy=result(1,2);
 s.value = result(1,:);
 C{1} = s;
 for i=2:size(result,1)
 added=false;

 for j=1:length(C)
 if ((abs(C{j}.minx - result(i,1)) < 10 || abs(C{j}.maxx - result(i,1)) < 10) && (abs(C{j}.miny - result(i,2)) < 10 || abs(C{j}.maxy - result(i,2)) < 10))
 % add to this bin
 added=true;
 C{j}.value(end+1,:) =result(i,:);
 if (result(i,1) < C{j}.minx); C{j}.minx = result(i,1); end
 if (result(i,1) > C{j}.maxx); C{j}.maxx = result(i,1); end
 if (result(i,2) < C{j}.miny); C{j}.miny = result(i,2); end
 if (result(i,2) > C{j}.maxy); C{j}.maxy = result(i,2); end
 break; % don't look anymore in C for bins
 end
 end
 if (added ~= true)
 s=struct;
 s.value=result(i,:);
 s.minx=result(i,1);
 s.maxx=result(i,1);
 s.miny=result(i,2);
 s.maxy=result(i,2);
 C{end+1}=s;
 end
 end
 final = zeros(length(C),4);
 for i=1:length(C)
 final(i,1)=C{i}.maxx;
 final(i,2)=C{i}.maxy;
 final(i,3)=mean(C{i}.value(:,3));
 final(i,4)=mean(C{i}.value(:,4));
 final(i,5)=mean(C{i}.value(:,5));
 final(i,6)=mean(C{i}.value(:,6));
 final(i,7)=mean(C{i}.value(:,7));
 end

 [~,ind]=sort(final(:,4)+final(:,5), 'descend');


 final = final(ind,:);
 dlmwrite(str,final);
 if (~isempty(list))
 dlmwrite([str '_scores'], scores);
 dlmwrite([str '_control_scores'], control_scores);
 end

 function [result, scores, control_scores] = call_blockbuster(M, p_var, p_sign, list, control)
 if (nargin < 4)
 list=[];
 end
 result=[];
 scores=[];
 control_scores=[];

 endpt = size(M,1);
 for limstart=1:1000:endpt
 limend = limstart+2000;
 if (limend > endpt)
 limend = endpt;
 end
 if (~isempty(list))
 lst = list(list(:,1) >= limstart & list(:,1) <= limend & list(:,2) >= limstart & list(:,2) <= limend & list(:,3) >= limstart & list(:,3) <= limend & list(:,4) >= limstart & list(:,4) <= limend, :);
 lst1 = control(control(:,1) >= limstart & control(:,1) <= limend & control(:,2) >= limstart & control(:,2) <= limend & control(:,3) >= limstart & control(:,3) <= limend & control(:,4) >= limstart & control(:,4) <= limend, :);
 else
 lst=[];
 lst1=[];
 end

 [M1, sr, sr1]=blockbuster(full(M(limstart:limend,limstart:limend)),p_var,p_sign, lst-limstart+1, lst1-limstart+1);

 if (~isempty(M1))
 M1(:, 1:2)=M1(:,1:2)+limstart-1;
 end
 result=[result; M1];
 scores=[scores; lst, sr];
 control_scores=[control_scores; lst1, sr1];

 fprintf('.');
 end

 function res = merge_scores(scores)
 scores=sortrows(scores);
 for i=1:size(scores,1)
 if (isnan(scores(i,5)))
 scores(i,5)=-10000;
 end
 for j=i+1:size(scores,1)
 if (scores(i,1:4) == scores(j,1:4))
 scores(i,5)=nanmax(scores(i,5), scores(j,5));
 scores(j,5)=scores(i,5);
 end
 end
 end
 res=unique(scores, 'rows');
 res(res==-10000)=NaN;


 function [result, scores, scores1] = blockbuster(observed, var_thresh, sign_thresh, list, list1)
 % BLOCK_FINDER  Find blocks in input observed matrix.

 if (nargin < 4)
 list = [];
 list1=[];
 end
 disp_figures=0;
 %calculate D upstream, directionality index upstream
 preferred_window=length(observed);
 gap=7;
 D_upstream=0.*observed;
 for i=1:length(observed)
 window=min(length(observed)-i-gap,i-gap);
 window=min(window,preferred_window);
 A=fliplr(observed(i,i-window:i-gap));
 B=observed(i,i+gap:i+window);

 preference=(A-B)./(A+B);
 D_upstream(i,i+gap:i+window)=preference;
 end

 % calculate triangles
 [Up, Up_Sign, Up_Sq, Lo, Lo_Sign, Lo_Sq] = dyp_triangles(D_upstream);

 B1=(-Up+Lo)/max(max(-Up+Lo))+(-Up_Sign+Lo_Sign)/max(max(-Up_Sign+Lo_Sign))-(Up_Sq-Up.^2+Lo_Sq-Lo.^2)/max(max(Up_Sq-Up.^2+Lo_Sq-Lo.^2));

 scores = zeros(size(list,1), 1);
 for i=1:size(list,1)
 scores(i) = max(max(B1(list(i,1):list(i,2), list(i,3):list(i,4))));
 end
 scores1 = zeros(size(list1,1), 1);
 for i=1:size(list1,1)
 scores1(i) = max(max(B1(list1(i,1):list1(i,2), list1(i,3):list1(i,4))));
 end

 B1(-Up_Sign < sign_thresh) = 0;
 B1(Lo_Sign  < sign_thresh)  = 0;
 Uvar = Up_Sq-Up.^2;
 Lvar = Lo_Sq-Lo.^2;
 if (var_thresh ~= 1000)
 B1(Uvar+Lvar > var_thresh)=0;
 end

 % find connected components of local max and average them

 CC1 = bwconncomp(B1>0);
 result = zeros(CC1.NumObjects, 7);

 for i=1:CC1.NumObjects
 [I, J] = ind2sub(CC1.ImageSize, CC1.PixelIdxList{i});
 [score, ind] = max(B1(CC1.PixelIdxList{i}));
 result(i, 1:2) = [I(ind),J(ind)];
 result(i,3) = score;
 result(i,4) = Uvar(I(ind),J(ind));
 result(i,5) = Lvar(I(ind),J(ind));
 result(i,6) = -Up_Sign(I(ind),J(ind));
 result(i,7) = Lo_Sign(I(ind),J(ind));
 end

 if (disp_figures)
 white2=cat(3, ones(size(D_upstream)), ones(size(D_upstream)),ones(size(D_upstream)));
 white=cat(3, zeros(size(D_upstream)), zeros(size(D_upstream)),ones(size(D_upstream)));
 red_color_map= makeColorMap([1 1 1],[1 0 0],80);
 mymap=zeros(size(observed));
 for k=1:size(result,1)
 row1=max(result(k,1)-1,1);
 row2=min(result(k,1)+1,length(observed));
 col1=max(result(k,2)-1,1);
 col2=min(result(k,2)+1,length(observed));

 mymap(row1:row2, row1:col2)=1;
 mymap(row1:col2, col1:col2)=1;
 end
 tmp=observed;  tmp(tmp>100)=100;
 h4=figure; imshow(white, 'InitialMagnification', 300); axis('square'); hold on; h=imagesc(make_symmetric(tmp)); hold off;
 set(h4, 'Position', [0 0 2000 2000]);
 colormap(red_color_map);
 set(h, 'AlphaData',mymap==0);
 h4=figure; imshow(white, 'InitialMagnification', 300); axis('square'); hold on; h=imagesc(make_symmetric(D_upstream)); hold off;
 set(h4, 'Position', [0 0 2000 2000]);
 h4=figure; h=imshow(white2, 'InitialMagnification', 300); axis('square'); %hold on; h=imshow(make_symmetric(B1)); hold off;
 set(h4, 'Position', [0 0 2000 2000]);
 set(h, 'AlphaData',B1~=0);
 %         h4=figure; imshow(white, 'InitialMagnification', 300); axis('square'); hold on; h=imagesc(make_symmetric(Uvar)); hold off;
 %         set(h4, 'Position', [0 0 2000 2000]);
 %         h4=figure; imshow(white, 'InitialMagnification', 300); axis('square'); hold on; h=imagesc(make_symmetric(Lvar)); hold off;
 %         set(h4, 'Position', [0 0 2000 2000]);
 %          h4=figure; imshow(white, 'InitialMagnification', 300); axis('square'); hold on; h=imagesc(make_symmetric(-Up_Sign)); hold off;
 %         set(h4, 'Position', [0 0 2000 2000]);
 %         h4=figure; imshow(white, 'InitialMagnification', 300); axis('square'); hold on; h=imagesc(make_symmetric(Lo_Sign)); hold off;
 %         set(h4, 'Position', [0 0 2000 2000]);
 %          h4=figure; imshow(white, 'InitialMagnification', 300); axis('square'); hold on; h=imagesc(make_symmetric(-Up)); hold off;
 %         set(h4, 'Position', [0 0 2000 2000]);
 %         h4=figure; imshow(white, 'InitialMagnification', 300); axis('square'); hold on; h=imagesc(make_symmetric(Lo)); hold off;
 %         set(h4, 'Position', [0 0 2000 2000]);
 end
 end

 function Y = make_symmetric(X)
 Y = X;
 for j=2:length(X)
 for i=1:j-1
 Y(j,i)=X(i,j);
 end
 end
 end

 function [Up, Up_Sign, Up_Sq, Lo, Lo_Sign, Lo_Sq] = dyp_triangles(M)

 %calculate Bnew, the block score matrix. it's a combination of 3 matrices
 M(isnan(M))=0;
 window=length(M); % not using this because it messed things up

 % Matrices used as dynamic programming lookups.
 % "R" matrices are sums of the columns up to that point: R(1,5) is sum of
 % column 5 from diagonal (row 5) up to row 1
 % "U" matrices are sums of the rows up to the point: U(1,5) is sum of row 5
 % from diagonal (col 1) up to col 5
 % We want mean, mean of sign, and variance, so we are doing the sum then
 % dividing by counts
 Rsum=dyp_right(M, window);
 Rsign=dyp_right(sign(M), window);
 Rsq=dyp_right(M.*M, window);
 Rcount=dyp_right(ones(size(M)), length(M));

 Usum=dyp_upper(M,window);
 Usign=dyp_upper(sign(M), window);
 Usq=dyp_upper(M.*M,window);
 Ucount=dyp_upper(ones(size(M)), length(M));

 Up=0.*M;
 Up_Sign=0.*M;
 Up_Sq=0.*M;
 Up_Count=0.*M;
 for i=1:length(Up)
 for j=i+1:length(Up)
 bot = floor((j-i+1)/2);
 % add half of column
 Up(i,j)= Up(i,j-1) + Rsum(i,j) - Rsum(i+bot, j);
 Up_Sign(i,j) = Up_Sign(i,j-1) + Rsign(i,j) - Rsign(i+bot, j);
 Up_Sq(i,j) = Up_Sq(i,j-1) + Rsq(i,j) - Rsq(i+bot, j);
 Up_Count(i,j)= Up_Count(i,j-1) + Rcount(i,j) - Rcount(i+bot, j);
 end
 end

 % Normalize
 Up_Count(Up_Count==0)=1;
 Up=Up./Up_Count;
 Up_Sign=Up_Sign./Up_Count;
 Up_Sq=Up_Sq./Up_Count;

 % Lower triangle
 Lo=0.*M;
 Lo_Sign=0.*M;
 Lo_Sq=0.*M;
 Lo_Count=0.*M;
 for a=1:length(Lo)
 for b=a+1:length(Lo)
 val=floor((b-a+1)/2);
 endpt = min(2*b-a,length(Lo));
 Lo_Count(a,b)=Lo_Count(a,b-1)+Ucount(b,endpt)-Rcount(a+val,b);
 Lo(a,b)=Lo(a,b-1)+Usum(b,endpt)-Rsum(a+val,b);
 Lo_Sign(a,b)=Lo_Sign(a,b-1)+Usign(b,endpt)-Rsign(a+val,b);
 Lo_Sq(a,b)=Lo_Sq(a,b-1)+Usq(b,endpt)-Rsq(a+val,b);
 end
 end
 Lo_Count(Lo_Count==0)=1;
 Lo=Lo./Lo_Count;
 Lo_Sign=Lo_Sign./Lo_Count;
 Lo_Sq=Lo_Sq./Lo_Count;
 end

 function M = flip_antidiagonal(N)
 M=N(end:-1:1,:);
 M=fliplr(M);
 M=M';
 end

 function U = dyp_upper(M, maxsize)
 % DYP_UPPER Dynamic programming to calculate "upper" matrix
 %   Initialize by setting the diagonal to the diagonal of original
 %   Iterate down (for each row) and to the left.
 v=diag(M);
 U=diag(v);
 for i=1:length(U)
 endpoint=i+1+maxsize;
 if (endpoint > length(U))
 endpoint = length(U);
 end
 for j=i+1:endpoint
 U(i,j)=U(i,j-1)+M(i,j);
 end
 end
 end

 function R = dyp_right(M, maxsize)
 % DYP_RIGHT Dynamic programming to calculate "right" matrix
 %   Initialize by setting the diagonal to the diagonal of original
 %   Iterate to the right and up.
 v=diag(M);
 R=diag(v);
 % j is column, i is row
 for j=2:length(R)
 endpoint=j-1-maxsize;
 if (endpoint < 1)
 endpoint = 1;
 end
 for i=j-1:-1:endpoint
 R(i,j)=M(i,j)+R(i+1,j);
 end
 end
 end

 function S = dyp_sum(M, super)
 v=diag(M);
 if (super > 1)
 S=zeros(size(M));
 else
 S=diag(v);
 end
 % d = distance from diagonal
 for d=super:length(S)-1
 % i = row, column is i+d
 % result is entry to left + entry below + orig value - diagonal
 % down (else we double count)
 for i=1:length(S)-d
 S(i,i+d)=S(i,i+d-1)+S(i+1,i+d)+M(i,i+d)-S(i+1,i+d-1);
 end
 end
 end


 */


}

