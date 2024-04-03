classdef AnalyzeWavePacket < handle
   %
   % This file is part of pySpec
   % Copyright (C) 2024  Luis Domenianni
   %
   % This program is free software: you can redistribute it and/or modify
   % it under the terms of the GNU General Public License as published by
   % the Free Software Foundation, either version 3 of the License, or
   % (at your option) any later version.
   %
   % pySpec is distributed in the hope that it will be useful,
   % but WITHOUT ANY WARRANTY; without even the implied warranty of
   % MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   % GNU General Public License for more details.
   %
   % You should have received a copy of the GNU General Public License
   % along with this program.  If not, see <https://www.gnu.org/licenses/>.
   %
    
   properties
       wavelength
       delay
       DeltaOD
       order
       startDelayIndex
       results
   end

   methods
       function results = getResults(this, idx)
           results = this.results{idx};
       end

       function HSVD(this, idx, order)

           % Adapted from Barkhuijsen et al. JMR, vol. 73, 553-557, 1987
           % Delay step, all increments must be even
           delayStep = this.delay(end)-this.delay(end-1);
           
           % Delay of first point
           delay0 = this.delay(this.startDelayIndex);
           
           % Get data
           data = this.DeltaOD(idx, this.startDelayIndex:end);
           
           % generate the Hankel matrix of LxN
           % where L + N = length(data)
           % and 0.5 =< L/N (=1.25) =< 2.0 for optimal results.
                    
           L        = round(1.25/2.25*length(data));
           colFirst = data(1:length(data)-L+1);
           rowLast  = data(length(data)-L+1:length(data));
           H        = hankel(colFirst,rowLast);

           % Singular Value decomposition of the Hankel matrix
           [U,~,~] = svd(H);
           
           % take the truncated matrix and calculate the Z matrix 
           % 'Matrix Computations' by Golub and van Loan, page 3,
           % for the Sherman-Morrison inversion formula.

           u = U(length(data)-L+1,1:order)';
           Z = eye(order)+u*u'/(1-u'*u);
           Z = Z*U(1:length(data)-L,1:order)'*U(2:length(data)-L+1,1:order);
                      
           % calculate signal poles by eigen-decomposition of Z

           poles = eig(Z);
           clear Z

           % calculate the frequencies and time-constants
           frequency=angle(poles)/(2*pi*delayStep);
           ratecon=-log(abs(poles))/delayStep;
           delayConstant=1 ./ratecon;
	
           % now set up the system of equations to calculate the
           % amplitudes and phases...
           % first the matrix
           Z=poles*poles';
           Z=(Z.^(length(data))-1)./(Z-1);

           % now the vector to speed up the routine we use the matlab
           % function 'cumprod' to create a matrix of
           % ascending powers of the conjugate of the
           % vector 'poles'
           vmat=ones(order,length(data));
           vmat(:,2)=conj(poles);

           X=cumprod(vmat');
           X1=cumprod(X);
           vmat =X1';
           clear X X1
           
           for j=1:order
                X=vmat(j,:);     
                rvect(j)=X*data';                                       
                clear X
           end

           % solve the linear system
           svect=(rvect/Z)';

           % calculate the amplitudes and phases
           amplitude = abs(svect).*exp(delay0./delayConstant);
           phase = rem(-(angle(svect)+2*pi*frequency*delay0)*360/(2*pi),360);

           results = [amplitude frequency delayConstant phase];
           [~,I] = sort(abs(results(:,2)));
           results = results(I,:);
           this.results = insert(this.results, idx, {results});
       end
       
       function [delayFit, dataFit] = fitData(this, idx)
           parameter = this.results{idx};

           delayFit = this.delay(this.startDelayIndex:end);
           dataFit = zeros(length(delayFit),1);

           for j = 1:size( parameter, 1 );
               dataFit = dataFit + ( parameter(j,1) * cos( 2 * pi * parameter(j,2) * delayFit +                      ...
                                                           parameter(j,4)*(2*pi)/360)                                ...
                                  .* exp( -delayFit / parameter(j,3) ))'                                               ;
           end
       end
       
       function lowPass(this, idx, freqCutOff)
           indices = find(abs(this.results{idx}(:,2)) > freqCutOff);
           this.results{idx}(indices,:) = [];
       end

       function highPass(this, idx, freqCutOff)
           indices = find(abs(this.results{idx}(:,2)) < freqCutOff);
           this.results{idx}(indices,:) = [];
       end
       end
   
   % Constructor
   methods
       function this = AnalyzeWavePacket(wavelength,...
                                         delay,...
                                         DeltaOD,...
                                         startDelayIndex)
                                    
       this.wavelength          = double(wavelength);
       this.delay               = double(delay);
       this.DeltaOD             = double(DeltaOD);
       this.startDelayIndex     = startDelayIndex;
       this.results             = dictionary();
       
       if isrow(this.wavelength)
           this.wavelength = this.wavelength';
       end
       if iscolumn(this.delay)
           this.delay = this.delay';
       end
       
       if ~isequal(size(this.DeltaOD), [length(this.wavelength) length(this.delay)])
           if isequal(size(this.DeltaOD), [length(this.delay) length(this.wavelength)])
               this.DeltaOD=this.DeltaOD';
           else
               warning('something wrong with DOD matrix')
           end
       end
       end
   end
end
