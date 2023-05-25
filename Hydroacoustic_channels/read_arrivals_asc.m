function [ amp, delay, SrcAngle, RcvrAngle, NumTopBnc, NumBotBnc, narrmat, Pos ] ...
    = read_arrivals_asc( ARRFIL, narrmx )

% read_arrivals_asc
% useage:
%[ amp, delay, SrcAngle, RcvrAngle, NumTopBnc, NumBotBnc, narrmat, Pos ] ...
%    = read_arrivals_asc( ARRFIL, narrmx );
%
% Loads the arrival time/amplitude data computed by BELLHOP
% You must set the string ARRFIL specifying the Arrivals File
% narrmx specifies the maximum number of arrivals allowed
% mbp 9/96

if nargin == 1
    narrmx = 200;
end

fid = fopen( ARRFIL, 'r');	% open the file
if ( fid == -1 )
    errordlg( 'read_arrivals_asc: Arrivals file cannot be opened' )
end

% read the header info

freq = fscanf( fid, '%f',  1  );
Nsd  = fscanf( fid, '%i',  1  );
Nrd  = fscanf( fid, '%i',  1  );
Nrr  = fscanf( fid, '%i',  1  );
Pos.s.depth   = fscanf( fid, '%f', Nsd );
Pos.r.depth   = fscanf( fid, '%f', Nrd );
Pos.r.range   = fscanf( fid, '%f', Nrr  );

% loop to read all the arrival info (delay and amplitude)

amp       = zeros( Nrr, narrmx, Nrd, Nsd );
delay     = zeros( Nrr, narrmx, Nrd, Nsd );
SrcAngle  = zeros( Nrr, narrmx, Nrd, Nsd );
RcvrAngle = zeros( Nrr, narrmx, Nrd, Nsd );
NumTopBnc = zeros( Nrr, narrmx, Nrd, Nsd );
NumBotBnc = zeros( Nrr, narrmx, Nrd, Nsd );

for isd = 1:Nsd
  narrmx2 = fscanf( fid, '%i', 1 )  % max. number of arrivals to follow
  
  for ird = 1:Nrd
    for ir = 1:Nrr
      narr = fscanf( fid, '%i', 1 );	% number of arrivals
      narrmat( ir, ird, isd ) = narr;

      if narr > 0   % do we have any arrivals?
        da = fscanf( fid, '%f', [ 7, narr ] );
        narr = min( narr, narrmx ); % we'll keep no more than narrmx values
        narrmat( ir, ird, isd ) = narr;
        
%        if ird == iphone   % take data for one phone only
          amp(       ir, 1:narr, ird, isd ) = da( 1, : ) .* exp( i * da( 2, : ) * pi/180);
          delay(     ir, 1:narr, ird, isd ) = da( 3, : );
          SrcAngle(  ir, 1:narr, ird, isd ) = da( 4, : );
          RcvrAngle( ir, 1:narr, ird, isd ) = da( 5, : );
          NumTopBnc( ir, 1:narr, ird, isd ) = da( 6, : );
          NumBotBnc( ir, 1:narr, ird, isd ) = da( 7, : );
%        end
      end
    end		% next receiver range
  end		% next receiver depth
end	% next source depth

fclose( fid );
