ny = 256; //128;//
nx = 128; //256;
ly = Sqrt(5);
lx = 0.5*ly;

//lx = 1;
//ly = 1;

Point(1) = { 0,  0, 0};
Point(2) = {lx,  0, 0};
Point(3) = {lx, ly, 0};
Point(4) = { 0, ly, 0};

Line(1) = {1,2};
Line(2) = {2,3};
Line(3) = {3,4};
Line(4) = {4,1};

Line Loop(1) = {1,2,3,4};
Ruled Surface(1) = {1};
Transfinite Surface(1) = {1,2,3,4};

Recombine Surface(1);
Transfinite Line{1,3} = nx+1;
Transfinite Line{2,4} = ny+1;

//Physical Line(1) = {1,2,3,4};
Physical Surface(10) = {1};

Physical Line(1) = {1}; // bottom
Physical Line(2) = {2}; // right
Physical Line(3) = {3}; // top
Physical Line(4) = {4}; // left
   
