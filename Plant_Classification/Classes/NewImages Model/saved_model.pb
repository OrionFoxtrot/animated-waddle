Що
п(Д(
D
AddV2
x"T
y"T
z"T"
Ttype:
2	АР
E
AssignAddVariableOp
resource
value"dtype"
dtypetypeИ
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( И
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
Ы
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

└
Conv2DBackpropInput
input_sizes
filter"T
out_backprop"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

Л
DepthwiseConv2dNative

input"T
filter"T
output"T"
Ttype:
2"
strides	list(int)",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

9
DivNoNan
x"T
y"T
z"T"
Ttype:

2
R
Equal
x"T
y"T
z
"	
Ttype"$
incompatible_shape_errorbool(Р
о
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
=
Greater
x"T
y"T
z
"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
,
Log
x"T
y"T"
Ttype:

2
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
В
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
>
Maximum
x"T
y"T
z"T"
Ttype:
2	
Н
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(И
>
Minimum
x"T
y"T
z"T"
Ttype:
2	
?
Mul
x"T
y"T
z"T"
Ttype:
2	Р
0
Neg
x"T
y"T"
Ttype:
2
	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
Н
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
Щ
ResizeNearestNeighbor
images"T
size
resized_images"T"
Ttype:
2
	"
align_cornersbool( "
half_pixel_centersbool( 
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
O
Size

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
┴
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring Ии
@
StaticRegexFullMatch	
input

output
"
patternstring
ў
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
М
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.9.12v2.9.0-18-gd8ce9f9c3018╨┬
А
Adam/my_dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/my_dense/bias/v
y
(Adam/my_dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/my_dense/bias/v*
_output_shapes
:*
dtype0
И
Adam/my_dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/my_dense/kernel/v
Б
*Adam/my_dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/my_dense/kernel/v*
_output_shapes

:*
dtype0
Ф
Adam/conv2d_transpose_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name Adam/conv2d_transpose_1/bias/v
Н
2Adam/conv2d_transpose_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_1/bias/v*
_output_shapes
:*
dtype0
д
 Adam/conv2d_transpose_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/conv2d_transpose_1/kernel/v
Э
4Adam/conv2d_transpose_1/kernel/v/Read/ReadVariableOpReadVariableOp Adam/conv2d_transpose_1/kernel/v*&
_output_shapes
:*
dtype0
Р
Adam/conv2d_transpose/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nameAdam/conv2d_transpose/bias/v
Й
0Adam/conv2d_transpose/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose/bias/v*
_output_shapes
:*
dtype0
а
Adam/conv2d_transpose/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name Adam/conv2d_transpose/kernel/v
Щ
2Adam/conv2d_transpose/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose/kernel/v*&
_output_shapes
:*
dtype0
Ф
Adam/separable_conv2d_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name Adam/separable_conv2d_3/bias/v
Н
2Adam/separable_conv2d_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/separable_conv2d_3/bias/v*
_output_shapes
:*
dtype0
╕
*Adam/separable_conv2d_3/pointwise_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*;
shared_name,*Adam/separable_conv2d_3/pointwise_kernel/v
▒
>Adam/separable_conv2d_3/pointwise_kernel/v/Read/ReadVariableOpReadVariableOp*Adam/separable_conv2d_3/pointwise_kernel/v*&
_output_shapes
:*
dtype0
╕
*Adam/separable_conv2d_3/depthwise_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*;
shared_name,*Adam/separable_conv2d_3/depthwise_kernel/v
▒
>Adam/separable_conv2d_3/depthwise_kernel/v/Read/ReadVariableOpReadVariableOp*Adam/separable_conv2d_3/depthwise_kernel/v*&
_output_shapes
:*
dtype0
Ф
Adam/separable_conv2d_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name Adam/separable_conv2d_2/bias/v
Н
2Adam/separable_conv2d_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/separable_conv2d_2/bias/v*
_output_shapes
:*
dtype0
╕
*Adam/separable_conv2d_2/pointwise_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*;
shared_name,*Adam/separable_conv2d_2/pointwise_kernel/v
▒
>Adam/separable_conv2d_2/pointwise_kernel/v/Read/ReadVariableOpReadVariableOp*Adam/separable_conv2d_2/pointwise_kernel/v*&
_output_shapes
:*
dtype0
╕
*Adam/separable_conv2d_2/depthwise_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*;
shared_name,*Adam/separable_conv2d_2/depthwise_kernel/v
▒
>Adam/separable_conv2d_2/depthwise_kernel/v/Read/ReadVariableOpReadVariableOp*Adam/separable_conv2d_2/depthwise_kernel/v*&
_output_shapes
:*
dtype0
Ф
Adam/separable_conv2d_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name Adam/separable_conv2d_1/bias/v
Н
2Adam/separable_conv2d_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/separable_conv2d_1/bias/v*
_output_shapes
:*
dtype0
╕
*Adam/separable_conv2d_1/pointwise_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*;
shared_name,*Adam/separable_conv2d_1/pointwise_kernel/v
▒
>Adam/separable_conv2d_1/pointwise_kernel/v/Read/ReadVariableOpReadVariableOp*Adam/separable_conv2d_1/pointwise_kernel/v*&
_output_shapes
:*
dtype0
╕
*Adam/separable_conv2d_1/depthwise_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*;
shared_name,*Adam/separable_conv2d_1/depthwise_kernel/v
▒
>Adam/separable_conv2d_1/depthwise_kernel/v/Read/ReadVariableOpReadVariableOp*Adam/separable_conv2d_1/depthwise_kernel/v*&
_output_shapes
:*
dtype0
Р
Adam/separable_conv2d/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nameAdam/separable_conv2d/bias/v
Й
0Adam/separable_conv2d/bias/v/Read/ReadVariableOpReadVariableOpAdam/separable_conv2d/bias/v*
_output_shapes
:*
dtype0
┤
(Adam/separable_conv2d/pointwise_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*9
shared_name*(Adam/separable_conv2d/pointwise_kernel/v
н
<Adam/separable_conv2d/pointwise_kernel/v/Read/ReadVariableOpReadVariableOp(Adam/separable_conv2d/pointwise_kernel/v*&
_output_shapes
:*
dtype0
┤
(Adam/separable_conv2d/depthwise_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*9
shared_name*(Adam/separable_conv2d/depthwise_kernel/v
н
<Adam/separable_conv2d/depthwise_kernel/v/Read/ReadVariableOpReadVariableOp(Adam/separable_conv2d/depthwise_kernel/v*&
_output_shapes
:*
dtype0
А
Adam/my_dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/my_dense/bias/m
y
(Adam/my_dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/my_dense/bias/m*
_output_shapes
:*
dtype0
И
Adam/my_dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/my_dense/kernel/m
Б
*Adam/my_dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/my_dense/kernel/m*
_output_shapes

:*
dtype0
Ф
Adam/conv2d_transpose_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name Adam/conv2d_transpose_1/bias/m
Н
2Adam/conv2d_transpose_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_1/bias/m*
_output_shapes
:*
dtype0
д
 Adam/conv2d_transpose_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/conv2d_transpose_1/kernel/m
Э
4Adam/conv2d_transpose_1/kernel/m/Read/ReadVariableOpReadVariableOp Adam/conv2d_transpose_1/kernel/m*&
_output_shapes
:*
dtype0
Р
Adam/conv2d_transpose/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nameAdam/conv2d_transpose/bias/m
Й
0Adam/conv2d_transpose/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose/bias/m*
_output_shapes
:*
dtype0
а
Adam/conv2d_transpose/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name Adam/conv2d_transpose/kernel/m
Щ
2Adam/conv2d_transpose/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose/kernel/m*&
_output_shapes
:*
dtype0
Ф
Adam/separable_conv2d_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name Adam/separable_conv2d_3/bias/m
Н
2Adam/separable_conv2d_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/separable_conv2d_3/bias/m*
_output_shapes
:*
dtype0
╕
*Adam/separable_conv2d_3/pointwise_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*;
shared_name,*Adam/separable_conv2d_3/pointwise_kernel/m
▒
>Adam/separable_conv2d_3/pointwise_kernel/m/Read/ReadVariableOpReadVariableOp*Adam/separable_conv2d_3/pointwise_kernel/m*&
_output_shapes
:*
dtype0
╕
*Adam/separable_conv2d_3/depthwise_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*;
shared_name,*Adam/separable_conv2d_3/depthwise_kernel/m
▒
>Adam/separable_conv2d_3/depthwise_kernel/m/Read/ReadVariableOpReadVariableOp*Adam/separable_conv2d_3/depthwise_kernel/m*&
_output_shapes
:*
dtype0
Ф
Adam/separable_conv2d_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name Adam/separable_conv2d_2/bias/m
Н
2Adam/separable_conv2d_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/separable_conv2d_2/bias/m*
_output_shapes
:*
dtype0
╕
*Adam/separable_conv2d_2/pointwise_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*;
shared_name,*Adam/separable_conv2d_2/pointwise_kernel/m
▒
>Adam/separable_conv2d_2/pointwise_kernel/m/Read/ReadVariableOpReadVariableOp*Adam/separable_conv2d_2/pointwise_kernel/m*&
_output_shapes
:*
dtype0
╕
*Adam/separable_conv2d_2/depthwise_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*;
shared_name,*Adam/separable_conv2d_2/depthwise_kernel/m
▒
>Adam/separable_conv2d_2/depthwise_kernel/m/Read/ReadVariableOpReadVariableOp*Adam/separable_conv2d_2/depthwise_kernel/m*&
_output_shapes
:*
dtype0
Ф
Adam/separable_conv2d_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name Adam/separable_conv2d_1/bias/m
Н
2Adam/separable_conv2d_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/separable_conv2d_1/bias/m*
_output_shapes
:*
dtype0
╕
*Adam/separable_conv2d_1/pointwise_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*;
shared_name,*Adam/separable_conv2d_1/pointwise_kernel/m
▒
>Adam/separable_conv2d_1/pointwise_kernel/m/Read/ReadVariableOpReadVariableOp*Adam/separable_conv2d_1/pointwise_kernel/m*&
_output_shapes
:*
dtype0
╕
*Adam/separable_conv2d_1/depthwise_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*;
shared_name,*Adam/separable_conv2d_1/depthwise_kernel/m
▒
>Adam/separable_conv2d_1/depthwise_kernel/m/Read/ReadVariableOpReadVariableOp*Adam/separable_conv2d_1/depthwise_kernel/m*&
_output_shapes
:*
dtype0
Р
Adam/separable_conv2d/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nameAdam/separable_conv2d/bias/m
Й
0Adam/separable_conv2d/bias/m/Read/ReadVariableOpReadVariableOpAdam/separable_conv2d/bias/m*
_output_shapes
:*
dtype0
┤
(Adam/separable_conv2d/pointwise_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*9
shared_name*(Adam/separable_conv2d/pointwise_kernel/m
н
<Adam/separable_conv2d/pointwise_kernel/m/Read/ReadVariableOpReadVariableOp(Adam/separable_conv2d/pointwise_kernel/m*&
_output_shapes
:*
dtype0
┤
(Adam/separable_conv2d/depthwise_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*9
shared_name*(Adam/separable_conv2d/depthwise_kernel/m
н
<Adam/separable_conv2d/depthwise_kernel/m/Read/ReadVariableOpReadVariableOp(Adam/separable_conv2d/depthwise_kernel/m*&
_output_shapes
:*
dtype0
t
add_metric/countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameadd_metric/count
m
$add_metric/count/Read/ReadVariableOpReadVariableOpadd_metric/count*
_output_shapes
: *
dtype0
t
add_metric/totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameadd_metric/total
m
$add_metric/total/Read/ReadVariableOpReadVariableOpadd_metric/total*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
r
my_dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namemy_dense/bias
k
!my_dense/bias/Read/ReadVariableOpReadVariableOpmy_dense/bias*
_output_shapes
:*
dtype0
z
my_dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namemy_dense/kernel
s
#my_dense/kernel/Read/ReadVariableOpReadVariableOpmy_dense/kernel*
_output_shapes

:*
dtype0
Ж
conv2d_transpose_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameconv2d_transpose_1/bias

+conv2d_transpose_1/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_1/bias*
_output_shapes
:*
dtype0
Ц
conv2d_transpose_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameconv2d_transpose_1/kernel
П
-conv2d_transpose_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_1/kernel*&
_output_shapes
:*
dtype0
В
conv2d_transpose/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameconv2d_transpose/bias
{
)conv2d_transpose/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose/bias*
_output_shapes
:*
dtype0
Т
conv2d_transpose/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameconv2d_transpose/kernel
Л
+conv2d_transpose/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose/kernel*&
_output_shapes
:*
dtype0
Ж
separable_conv2d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameseparable_conv2d_3/bias

+separable_conv2d_3/bias/Read/ReadVariableOpReadVariableOpseparable_conv2d_3/bias*
_output_shapes
:*
dtype0
к
#separable_conv2d_3/pointwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#separable_conv2d_3/pointwise_kernel
г
7separable_conv2d_3/pointwise_kernel/Read/ReadVariableOpReadVariableOp#separable_conv2d_3/pointwise_kernel*&
_output_shapes
:*
dtype0
к
#separable_conv2d_3/depthwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#separable_conv2d_3/depthwise_kernel
г
7separable_conv2d_3/depthwise_kernel/Read/ReadVariableOpReadVariableOp#separable_conv2d_3/depthwise_kernel*&
_output_shapes
:*
dtype0
Ж
separable_conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameseparable_conv2d_2/bias

+separable_conv2d_2/bias/Read/ReadVariableOpReadVariableOpseparable_conv2d_2/bias*
_output_shapes
:*
dtype0
к
#separable_conv2d_2/pointwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#separable_conv2d_2/pointwise_kernel
г
7separable_conv2d_2/pointwise_kernel/Read/ReadVariableOpReadVariableOp#separable_conv2d_2/pointwise_kernel*&
_output_shapes
:*
dtype0
к
#separable_conv2d_2/depthwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#separable_conv2d_2/depthwise_kernel
г
7separable_conv2d_2/depthwise_kernel/Read/ReadVariableOpReadVariableOp#separable_conv2d_2/depthwise_kernel*&
_output_shapes
:*
dtype0
Ж
separable_conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameseparable_conv2d_1/bias

+separable_conv2d_1/bias/Read/ReadVariableOpReadVariableOpseparable_conv2d_1/bias*
_output_shapes
:*
dtype0
к
#separable_conv2d_1/pointwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#separable_conv2d_1/pointwise_kernel
г
7separable_conv2d_1/pointwise_kernel/Read/ReadVariableOpReadVariableOp#separable_conv2d_1/pointwise_kernel*&
_output_shapes
:*
dtype0
к
#separable_conv2d_1/depthwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#separable_conv2d_1/depthwise_kernel
г
7separable_conv2d_1/depthwise_kernel/Read/ReadVariableOpReadVariableOp#separable_conv2d_1/depthwise_kernel*&
_output_shapes
:*
dtype0
В
separable_conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameseparable_conv2d/bias
{
)separable_conv2d/bias/Read/ReadVariableOpReadVariableOpseparable_conv2d/bias*
_output_shapes
:*
dtype0
ж
!separable_conv2d/pointwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!separable_conv2d/pointwise_kernel
Я
5separable_conv2d/pointwise_kernel/Read/ReadVariableOpReadVariableOp!separable_conv2d/pointwise_kernel*&
_output_shapes
:*
dtype0
ж
!separable_conv2d/depthwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!separable_conv2d/depthwise_kernel
Я
5separable_conv2d/depthwise_kernel/Read/ReadVariableOpReadVariableOp!separable_conv2d/depthwise_kernel*&
_output_shapes
:*
dtype0

NoOpNoOp
НМ
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*╟Л
value╝ЛB╕Л B░Л
№
layer-0
layer-1
layer-2
layer_with_weights-0
layer-3
layer_with_weights-1
layer-4
layer-5
layer_with_weights-2
layer-6
layer_with_weights-3
layer-7
	layer-8

layer_with_weights-4

layer-9
layer_with_weights-5
layer-10
layer_with_weights-6
layer-11
layer-12
layer-13
layer-14
layer-15
layer-16
layer-17
layer-18
layer-19
layer-20
layer-21
layer-22
layer-23
layer-24
layer-25
layer-26
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses
"_default_save_signature
#	optimizer
$loss
%
signatures*
* 
* 
* 
ш
&	variables
'trainable_variables
(regularization_losses
)	keras_api
*__call__
*+&call_and_return_all_conditional_losses
,depthwise_kernel
-pointwise_kernel
.bias
 /_jit_compiled_convolution_op*
ш
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses
6depthwise_kernel
7pointwise_kernel
8bias
 9_jit_compiled_convolution_op*
О
:	variables
;trainable_variables
<regularization_losses
=	keras_api
>__call__
*?&call_and_return_all_conditional_losses* 
ш
@	variables
Atrainable_variables
Bregularization_losses
C	keras_api
D__call__
*E&call_and_return_all_conditional_losses
Fdepthwise_kernel
Gpointwise_kernel
Hbias
 I_jit_compiled_convolution_op*
ш
J	variables
Ktrainable_variables
Lregularization_losses
M	keras_api
N__call__
*O&call_and_return_all_conditional_losses
Pdepthwise_kernel
Qpointwise_kernel
Rbias
 S_jit_compiled_convolution_op*
О
T	variables
Utrainable_variables
Vregularization_losses
W	keras_api
X__call__
*Y&call_and_return_all_conditional_losses* 
╚
Z	variables
[trainable_variables
\regularization_losses
]	keras_api
^__call__
*_&call_and_return_all_conditional_losses

`kernel
abias
 b_jit_compiled_convolution_op*
╚
c	variables
dtrainable_variables
eregularization_losses
f	keras_api
g__call__
*h&call_and_return_all_conditional_losses

ikernel
jbias
 k_jit_compiled_convolution_op*
ж
l	variables
mtrainable_variables
nregularization_losses
o	keras_api
p__call__
*q&call_and_return_all_conditional_losses

rkernel
sbias*
О
t	variables
utrainable_variables
vregularization_losses
w	keras_api
x__call__
*y&call_and_return_all_conditional_losses* 

z	keras_api* 

{	keras_api* 

|	keras_api* 

}	keras_api* 

~	keras_api* 

	keras_api* 
Ф
А	variables
Бtrainable_variables
Вregularization_losses
Г	keras_api
Д__call__
+Е&call_and_return_all_conditional_losses* 

Ж	keras_api* 

З	keras_api* 

И	keras_api* 

Й	keras_api* 

К	keras_api* 

Л	keras_api* 
Ц
М	variables
Нtrainable_variables
Оregularization_losses
П	keras_api
Р__call__
+С&call_and_return_all_conditional_losses*
К
,0
-1
.2
63
74
85
F6
G7
H8
P9
Q10
R11
`12
a13
i14
j15
r16
s17*
К
,0
-1
.2
63
74
85
F6
G7
H8
P9
Q10
R11
`12
a13
i14
j15
r16
s17*
* 
╡
Тnon_trainable_variables
Уlayers
Фmetrics
 Хlayer_regularization_losses
Цlayer_metrics
	variables
trainable_variables
regularization_losses
 __call__
"_default_save_signature
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses*
:
Чtrace_0
Шtrace_1
Щtrace_2
Ъtrace_3* 
:
Ыtrace_0
Ьtrace_1
Эtrace_2
Юtrace_3* 
* 
▒
	Яiter
аbeta_1
бbeta_2

вdecay
гlearning_rate,mГ-mД.mЕ6mЖ7mЗ8mИFmЙGmКHmЛPmМQmНRmО`mПamРimСjmТrmУsmФ,vХ-vЦ.vЧ6vШ7vЩ8vЪFvЫGvЬHvЭPvЮQvЯRvа`vбavвivгjvдrvеsvж*
* 

дserving_default* 

,0
-1
.2*

,0
-1
.2*
* 
Ш
еnon_trainable_variables
жlayers
зmetrics
 иlayer_regularization_losses
йlayer_metrics
&	variables
'trainable_variables
(regularization_losses
*__call__
*+&call_and_return_all_conditional_losses
&+"call_and_return_conditional_losses*

кtrace_0* 

лtrace_0* 
{u
VARIABLE_VALUE!separable_conv2d/depthwise_kernel@layer_with_weights-0/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUE!separable_conv2d/pointwise_kernel@layer_with_weights-0/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEseparable_conv2d/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

60
71
82*

60
71
82*
* 
Ш
мnon_trainable_variables
нlayers
оmetrics
 пlayer_regularization_losses
░layer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses*

▒trace_0* 

▓trace_0* 
}w
VARIABLE_VALUE#separable_conv2d_1/depthwise_kernel@layer_with_weights-1/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE#separable_conv2d_1/pointwise_kernel@layer_with_weights-1/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEseparable_conv2d_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
Ц
│non_trainable_variables
┤layers
╡metrics
 ╢layer_regularization_losses
╖layer_metrics
:	variables
;trainable_variables
<regularization_losses
>__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses* 

╕trace_0* 

╣trace_0* 

F0
G1
H2*

F0
G1
H2*
* 
Ш
║non_trainable_variables
╗layers
╝metrics
 ╜layer_regularization_losses
╛layer_metrics
@	variables
Atrainable_variables
Bregularization_losses
D__call__
*E&call_and_return_all_conditional_losses
&E"call_and_return_conditional_losses*

┐trace_0* 

└trace_0* 
}w
VARIABLE_VALUE#separable_conv2d_2/depthwise_kernel@layer_with_weights-2/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE#separable_conv2d_2/pointwise_kernel@layer_with_weights-2/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEseparable_conv2d_2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

P0
Q1
R2*

P0
Q1
R2*
* 
Ш
┴non_trainable_variables
┬layers
├metrics
 ─layer_regularization_losses
┼layer_metrics
J	variables
Ktrainable_variables
Lregularization_losses
N__call__
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses*

╞trace_0* 

╟trace_0* 
}w
VARIABLE_VALUE#separable_conv2d_3/depthwise_kernel@layer_with_weights-3/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE#separable_conv2d_3/pointwise_kernel@layer_with_weights-3/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEseparable_conv2d_3/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
Ц
╚non_trainable_variables
╔layers
╩metrics
 ╦layer_regularization_losses
╠layer_metrics
T	variables
Utrainable_variables
Vregularization_losses
X__call__
*Y&call_and_return_all_conditional_losses
&Y"call_and_return_conditional_losses* 

═trace_0* 

╬trace_0* 

`0
a1*

`0
a1*
* 
Ш
╧non_trainable_variables
╨layers
╤metrics
 ╥layer_regularization_losses
╙layer_metrics
Z	variables
[trainable_variables
\regularization_losses
^__call__
*_&call_and_return_all_conditional_losses
&_"call_and_return_conditional_losses*

╘trace_0* 

╒trace_0* 
ga
VARIABLE_VALUEconv2d_transpose/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEconv2d_transpose/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

i0
j1*

i0
j1*
* 
Ш
╓non_trainable_variables
╫layers
╪metrics
 ┘layer_regularization_losses
┌layer_metrics
c	variables
dtrainable_variables
eregularization_losses
g__call__
*h&call_and_return_all_conditional_losses
&h"call_and_return_conditional_losses*

█trace_0* 

▄trace_0* 
ic
VARIABLE_VALUEconv2d_transpose_1/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEconv2d_transpose_1/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

r0
s1*

r0
s1*
* 
Ш
▌non_trainable_variables
▐layers
▀metrics
 рlayer_regularization_losses
сlayer_metrics
l	variables
mtrainable_variables
nregularization_losses
p__call__
*q&call_and_return_all_conditional_losses
&q"call_and_return_conditional_losses*

тtrace_0* 

уtrace_0* 
_Y
VARIABLE_VALUEmy_dense/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEmy_dense/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
Ц
фnon_trainable_variables
хlayers
цmetrics
 чlayer_regularization_losses
шlayer_metrics
t	variables
utrainable_variables
vregularization_losses
x__call__
*y&call_and_return_all_conditional_losses
&y"call_and_return_conditional_losses* 

щtrace_0* 

ъtrace_0* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
Ь
ыnon_trainable_variables
ьlayers
эmetrics
 юlayer_regularization_losses
яlayer_metrics
А	variables
Бtrainable_variables
Вregularization_losses
Д__call__
+Е&call_and_return_all_conditional_losses
'Е"call_and_return_conditional_losses* 

Ёtrace_0* 

ёtrace_0* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
Ю
Єnon_trainable_variables
єlayers
Їmetrics
 їlayer_regularization_losses
Ўlayer_metrics
М	variables
Нtrainable_variables
Оregularization_losses
Р__call__
+С&call_and_return_all_conditional_losses
'С"call_and_return_conditional_losses*

ўtrace_0* 

°trace_0* 
* 
╥
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26*

∙0
·1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

·0*
* 

·weighted_accuracy*
* 
* 
<
√	variables
№	keras_api

¤total

■count*
<
 	variables
А	keras_api

Бtotal

Вcount*

¤0
■1*

√	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

Б0
В1*

 	variables*
^X
VARIABLE_VALUEadd_metric/total4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEadd_metric/count4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
ЯШ
VARIABLE_VALUE(Adam/separable_conv2d/depthwise_kernel/m\layer_with_weights-0/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ЯШ
VARIABLE_VALUE(Adam/separable_conv2d/pointwise_kernel/m\layer_with_weights-0/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ЗА
VARIABLE_VALUEAdam/separable_conv2d/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
бЪ
VARIABLE_VALUE*Adam/separable_conv2d_1/depthwise_kernel/m\layer_with_weights-1/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
бЪ
VARIABLE_VALUE*Adam/separable_conv2d_1/pointwise_kernel/m\layer_with_weights-1/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ЙВ
VARIABLE_VALUEAdam/separable_conv2d_1/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
бЪ
VARIABLE_VALUE*Adam/separable_conv2d_2/depthwise_kernel/m\layer_with_weights-2/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
бЪ
VARIABLE_VALUE*Adam/separable_conv2d_2/pointwise_kernel/m\layer_with_weights-2/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ЙВ
VARIABLE_VALUEAdam/separable_conv2d_2/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
бЪ
VARIABLE_VALUE*Adam/separable_conv2d_3/depthwise_kernel/m\layer_with_weights-3/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
бЪ
VARIABLE_VALUE*Adam/separable_conv2d_3/pointwise_kernel/m\layer_with_weights-3/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ЙВ
VARIABLE_VALUEAdam/separable_conv2d_3/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ЛД
VARIABLE_VALUEAdam/conv2d_transpose/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ЗА
VARIABLE_VALUEAdam/conv2d_transpose/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
НЖ
VARIABLE_VALUE Adam/conv2d_transpose_1/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ЙВ
VARIABLE_VALUEAdam/conv2d_transpose_1/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
В|
VARIABLE_VALUEAdam/my_dense/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/my_dense/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ЯШ
VARIABLE_VALUE(Adam/separable_conv2d/depthwise_kernel/v\layer_with_weights-0/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ЯШ
VARIABLE_VALUE(Adam/separable_conv2d/pointwise_kernel/v\layer_with_weights-0/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ЗА
VARIABLE_VALUEAdam/separable_conv2d/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
бЪ
VARIABLE_VALUE*Adam/separable_conv2d_1/depthwise_kernel/v\layer_with_weights-1/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
бЪ
VARIABLE_VALUE*Adam/separable_conv2d_1/pointwise_kernel/v\layer_with_weights-1/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ЙВ
VARIABLE_VALUEAdam/separable_conv2d_1/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
бЪ
VARIABLE_VALUE*Adam/separable_conv2d_2/depthwise_kernel/v\layer_with_weights-2/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
бЪ
VARIABLE_VALUE*Adam/separable_conv2d_2/pointwise_kernel/v\layer_with_weights-2/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ЙВ
VARIABLE_VALUEAdam/separable_conv2d_2/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
бЪ
VARIABLE_VALUE*Adam/separable_conv2d_3/depthwise_kernel/v\layer_with_weights-3/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
бЪ
VARIABLE_VALUE*Adam/separable_conv2d_3/pointwise_kernel/v\layer_with_weights-3/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ЙВ
VARIABLE_VALUEAdam/separable_conv2d_3/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ЛД
VARIABLE_VALUEAdam/conv2d_transpose/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ЗА
VARIABLE_VALUEAdam/conv2d_transpose/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
НЖ
VARIABLE_VALUE Adam/conv2d_transpose_1/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ЙВ
VARIABLE_VALUEAdam/conv2d_transpose_1/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
В|
VARIABLE_VALUEAdam/my_dense/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/my_dense/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Н
serving_default_img_inPlaceholder*1
_output_shapes
:         └└*
dtype0*&
shape:         └└
К
serving_default_true_labelsPlaceholder*-
_output_shapes
:         └└*
dtype0*"
shape:         └└
Й
serving_default_weights_inPlaceholder*-
_output_shapes
:         └└*
dtype0*"
shape:         └└
Т
StatefulPartitionedCallStatefulPartitionedCallserving_default_img_inserving_default_true_labelsserving_default_weights_in!separable_conv2d/depthwise_kernel!separable_conv2d/pointwise_kernelseparable_conv2d/bias#separable_conv2d_1/depthwise_kernel#separable_conv2d_1/pointwise_kernelseparable_conv2d_1/bias#separable_conv2d_2/depthwise_kernel#separable_conv2d_2/pointwise_kernelseparable_conv2d_2/bias#separable_conv2d_3/depthwise_kernel#separable_conv2d_3/pointwise_kernelseparable_conv2d_3/biasconv2d_transpose/kernelconv2d_transpose/biasconv2d_transpose_1/kernelconv2d_transpose_1/biasmy_dense/kernelmy_dense/biasadd_metric/totaladd_metric/count*"
Tin
2*

Tout
 *
_collective_manager_ids
 *4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *+
f&R$
"__inference_signature_wrapper_5425
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
╒
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename5separable_conv2d/depthwise_kernel/Read/ReadVariableOp5separable_conv2d/pointwise_kernel/Read/ReadVariableOp)separable_conv2d/bias/Read/ReadVariableOp7separable_conv2d_1/depthwise_kernel/Read/ReadVariableOp7separable_conv2d_1/pointwise_kernel/Read/ReadVariableOp+separable_conv2d_1/bias/Read/ReadVariableOp7separable_conv2d_2/depthwise_kernel/Read/ReadVariableOp7separable_conv2d_2/pointwise_kernel/Read/ReadVariableOp+separable_conv2d_2/bias/Read/ReadVariableOp7separable_conv2d_3/depthwise_kernel/Read/ReadVariableOp7separable_conv2d_3/pointwise_kernel/Read/ReadVariableOp+separable_conv2d_3/bias/Read/ReadVariableOp+conv2d_transpose/kernel/Read/ReadVariableOp)conv2d_transpose/bias/Read/ReadVariableOp-conv2d_transpose_1/kernel/Read/ReadVariableOp+conv2d_transpose_1/bias/Read/ReadVariableOp#my_dense/kernel/Read/ReadVariableOp!my_dense/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp$add_metric/total/Read/ReadVariableOp$add_metric/count/Read/ReadVariableOp<Adam/separable_conv2d/depthwise_kernel/m/Read/ReadVariableOp<Adam/separable_conv2d/pointwise_kernel/m/Read/ReadVariableOp0Adam/separable_conv2d/bias/m/Read/ReadVariableOp>Adam/separable_conv2d_1/depthwise_kernel/m/Read/ReadVariableOp>Adam/separable_conv2d_1/pointwise_kernel/m/Read/ReadVariableOp2Adam/separable_conv2d_1/bias/m/Read/ReadVariableOp>Adam/separable_conv2d_2/depthwise_kernel/m/Read/ReadVariableOp>Adam/separable_conv2d_2/pointwise_kernel/m/Read/ReadVariableOp2Adam/separable_conv2d_2/bias/m/Read/ReadVariableOp>Adam/separable_conv2d_3/depthwise_kernel/m/Read/ReadVariableOp>Adam/separable_conv2d_3/pointwise_kernel/m/Read/ReadVariableOp2Adam/separable_conv2d_3/bias/m/Read/ReadVariableOp2Adam/conv2d_transpose/kernel/m/Read/ReadVariableOp0Adam/conv2d_transpose/bias/m/Read/ReadVariableOp4Adam/conv2d_transpose_1/kernel/m/Read/ReadVariableOp2Adam/conv2d_transpose_1/bias/m/Read/ReadVariableOp*Adam/my_dense/kernel/m/Read/ReadVariableOp(Adam/my_dense/bias/m/Read/ReadVariableOp<Adam/separable_conv2d/depthwise_kernel/v/Read/ReadVariableOp<Adam/separable_conv2d/pointwise_kernel/v/Read/ReadVariableOp0Adam/separable_conv2d/bias/v/Read/ReadVariableOp>Adam/separable_conv2d_1/depthwise_kernel/v/Read/ReadVariableOp>Adam/separable_conv2d_1/pointwise_kernel/v/Read/ReadVariableOp2Adam/separable_conv2d_1/bias/v/Read/ReadVariableOp>Adam/separable_conv2d_2/depthwise_kernel/v/Read/ReadVariableOp>Adam/separable_conv2d_2/pointwise_kernel/v/Read/ReadVariableOp2Adam/separable_conv2d_2/bias/v/Read/ReadVariableOp>Adam/separable_conv2d_3/depthwise_kernel/v/Read/ReadVariableOp>Adam/separable_conv2d_3/pointwise_kernel/v/Read/ReadVariableOp2Adam/separable_conv2d_3/bias/v/Read/ReadVariableOp2Adam/conv2d_transpose/kernel/v/Read/ReadVariableOp0Adam/conv2d_transpose/bias/v/Read/ReadVariableOp4Adam/conv2d_transpose_1/kernel/v/Read/ReadVariableOp2Adam/conv2d_transpose_1/bias/v/Read/ReadVariableOp*Adam/my_dense/kernel/v/Read/ReadVariableOp(Adam/my_dense/bias/v/Read/ReadVariableOpConst*L
TinE
C2A	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *&
f!R
__inference__traced_save_6399
ф
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename!separable_conv2d/depthwise_kernel!separable_conv2d/pointwise_kernelseparable_conv2d/bias#separable_conv2d_1/depthwise_kernel#separable_conv2d_1/pointwise_kernelseparable_conv2d_1/bias#separable_conv2d_2/depthwise_kernel#separable_conv2d_2/pointwise_kernelseparable_conv2d_2/bias#separable_conv2d_3/depthwise_kernel#separable_conv2d_3/pointwise_kernelseparable_conv2d_3/biasconv2d_transpose/kernelconv2d_transpose/biasconv2d_transpose_1/kernelconv2d_transpose_1/biasmy_dense/kernelmy_dense/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountadd_metric/totaladd_metric/count(Adam/separable_conv2d/depthwise_kernel/m(Adam/separable_conv2d/pointwise_kernel/mAdam/separable_conv2d/bias/m*Adam/separable_conv2d_1/depthwise_kernel/m*Adam/separable_conv2d_1/pointwise_kernel/mAdam/separable_conv2d_1/bias/m*Adam/separable_conv2d_2/depthwise_kernel/m*Adam/separable_conv2d_2/pointwise_kernel/mAdam/separable_conv2d_2/bias/m*Adam/separable_conv2d_3/depthwise_kernel/m*Adam/separable_conv2d_3/pointwise_kernel/mAdam/separable_conv2d_3/bias/mAdam/conv2d_transpose/kernel/mAdam/conv2d_transpose/bias/m Adam/conv2d_transpose_1/kernel/mAdam/conv2d_transpose_1/bias/mAdam/my_dense/kernel/mAdam/my_dense/bias/m(Adam/separable_conv2d/depthwise_kernel/v(Adam/separable_conv2d/pointwise_kernel/vAdam/separable_conv2d/bias/v*Adam/separable_conv2d_1/depthwise_kernel/v*Adam/separable_conv2d_1/pointwise_kernel/vAdam/separable_conv2d_1/bias/v*Adam/separable_conv2d_2/depthwise_kernel/v*Adam/separable_conv2d_2/pointwise_kernel/vAdam/separable_conv2d_2/bias/v*Adam/separable_conv2d_3/depthwise_kernel/v*Adam/separable_conv2d_3/pointwise_kernel/vAdam/separable_conv2d_3/bias/vAdam/conv2d_transpose/kernel/vAdam/conv2d_transpose/bias/v Adam/conv2d_transpose_1/kernel/vAdam/conv2d_transpose_1/bias/vAdam/my_dense/kernel/vAdam/my_dense/bias/v*K
TinD
B2@*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *)
f$R"
 __inference__traced_restore_6598∙щ
░
H
,__inference_max_pooling2d_layer_call_fn_5932

inputs
identity╒
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_4548Г
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
░
H
,__inference_up_sampling2d_layer_call_fn_5996

inputs
identity╒
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_up_sampling2d_layer_call_and_return_conditional_losses_4625Г
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
к
Г
L__inference_separable_conv2d_1_layer_call_and_return_conditional_losses_5927

inputsB
(separable_conv2d_readvariableop_resource:D
*separable_conv2d_readvariableop_1_resource:-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвseparable_conv2d/ReadVariableOpв!separable_conv2d/ReadVariableOp_1Р
separable_conv2d/ReadVariableOpReadVariableOp(separable_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ф
!separable_conv2d/ReadVariableOp_1ReadVariableOp*separable_conv2d_readvariableop_1_resource*&
_output_shapes
:*
dtype0o
separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            o
separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      ╪
separable_conv2d/depthwiseDepthwiseConv2dNativeinputs'separable_conv2d/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           *
paddingSAME*
strides
▀
separable_conv2dConv2D#separable_conv2d/depthwise:output:0)separable_conv2d/ReadVariableOp_1:value:0*
T0*A
_output_shapes/
-:+                           *
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0Щ
BiasAddBiasAddseparable_conv2d:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           j
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+                           {
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+                           е
NoOpNoOp^BiasAdd/ReadVariableOp ^separable_conv2d/ReadVariableOp"^separable_conv2d/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:+                           : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
separable_conv2d/ReadVariableOpseparable_conv2d/ReadVariableOp2F
!separable_conv2d/ReadVariableOp_1!separable_conv2d/ReadVariableOp_1:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
щ■
с
__inference__wrapped_model_4481

img_in

weights_in
true_labels[
Amodel_1_separable_conv2d_separable_conv2d_readvariableop_resource:]
Cmodel_1_separable_conv2d_separable_conv2d_readvariableop_1_resource:F
8model_1_separable_conv2d_biasadd_readvariableop_resource:]
Cmodel_1_separable_conv2d_1_separable_conv2d_readvariableop_resource:_
Emodel_1_separable_conv2d_1_separable_conv2d_readvariableop_1_resource:H
:model_1_separable_conv2d_1_biasadd_readvariableop_resource:]
Cmodel_1_separable_conv2d_2_separable_conv2d_readvariableop_resource:_
Emodel_1_separable_conv2d_2_separable_conv2d_readvariableop_1_resource:H
:model_1_separable_conv2d_2_biasadd_readvariableop_resource:]
Cmodel_1_separable_conv2d_3_separable_conv2d_readvariableop_resource:_
Emodel_1_separable_conv2d_3_separable_conv2d_readvariableop_1_resource:H
:model_1_separable_conv2d_3_biasadd_readvariableop_resource:[
Amodel_1_conv2d_transpose_conv2d_transpose_readvariableop_resource:F
8model_1_conv2d_transpose_biasadd_readvariableop_resource:]
Cmodel_1_conv2d_transpose_1_conv2d_transpose_readvariableop_resource:H
:model_1_conv2d_transpose_1_biasadd_readvariableop_resource:D
2model_1_my_dense_tensordot_readvariableop_resource:>
0model_1_my_dense_biasadd_readvariableop_resource:9
/model_1_add_metric_assignaddvariableop_resource: ;
1model_1_add_metric_assignaddvariableop_1_resource: Ив&model_1/add_metric/AssignAddVariableOpв(model_1/add_metric/AssignAddVariableOp_1в,model_1/add_metric/div_no_nan/ReadVariableOpв.model_1/add_metric/div_no_nan/ReadVariableOp_1в/model_1/conv2d_transpose/BiasAdd/ReadVariableOpв8model_1/conv2d_transpose/conv2d_transpose/ReadVariableOpв1model_1/conv2d_transpose_1/BiasAdd/ReadVariableOpв:model_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOpв'model_1/my_dense/BiasAdd/ReadVariableOpв)model_1/my_dense/Tensordot/ReadVariableOpв/model_1/separable_conv2d/BiasAdd/ReadVariableOpв8model_1/separable_conv2d/separable_conv2d/ReadVariableOpв:model_1/separable_conv2d/separable_conv2d/ReadVariableOp_1в1model_1/separable_conv2d_1/BiasAdd/ReadVariableOpв:model_1/separable_conv2d_1/separable_conv2d/ReadVariableOpв<model_1/separable_conv2d_1/separable_conv2d/ReadVariableOp_1в1model_1/separable_conv2d_2/BiasAdd/ReadVariableOpв:model_1/separable_conv2d_2/separable_conv2d/ReadVariableOpв<model_1/separable_conv2d_2/separable_conv2d/ReadVariableOp_1в1model_1/separable_conv2d_3/BiasAdd/ReadVariableOpв:model_1/separable_conv2d_3/separable_conv2d/ReadVariableOpв<model_1/separable_conv2d_3/separable_conv2d/ReadVariableOp_1┬
8model_1/separable_conv2d/separable_conv2d/ReadVariableOpReadVariableOpAmodel_1_separable_conv2d_separable_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0╞
:model_1/separable_conv2d/separable_conv2d/ReadVariableOp_1ReadVariableOpCmodel_1_separable_conv2d_separable_conv2d_readvariableop_1_resource*&
_output_shapes
:*
dtype0И
/model_1/separable_conv2d/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            И
7model_1/separable_conv2d/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      ·
3model_1/separable_conv2d/separable_conv2d/depthwiseDepthwiseConv2dNativeimg_in@model_1/separable_conv2d/separable_conv2d/ReadVariableOp:value:0*
T0*1
_output_shapes
:         └└*
paddingSAME*
strides
Ъ
)model_1/separable_conv2d/separable_conv2dConv2D<model_1/separable_conv2d/separable_conv2d/depthwise:output:0Bmodel_1/separable_conv2d/separable_conv2d/ReadVariableOp_1:value:0*
T0*1
_output_shapes
:         └└*
paddingVALID*
strides
д
/model_1/separable_conv2d/BiasAdd/ReadVariableOpReadVariableOp8model_1_separable_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0╘
 model_1/separable_conv2d/BiasAddBiasAdd2model_1/separable_conv2d/separable_conv2d:output:07model_1/separable_conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         └└М
model_1/separable_conv2d/ReluRelu)model_1/separable_conv2d/BiasAdd:output:0*
T0*1
_output_shapes
:         └└╞
:model_1/separable_conv2d_1/separable_conv2d/ReadVariableOpReadVariableOpCmodel_1_separable_conv2d_1_separable_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0╩
<model_1/separable_conv2d_1/separable_conv2d/ReadVariableOp_1ReadVariableOpEmodel_1_separable_conv2d_1_separable_conv2d_readvariableop_1_resource*&
_output_shapes
:*
dtype0К
1model_1/separable_conv2d_1/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            К
9model_1/separable_conv2d_1/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      г
5model_1/separable_conv2d_1/separable_conv2d/depthwiseDepthwiseConv2dNative+model_1/separable_conv2d/Relu:activations:0Bmodel_1/separable_conv2d_1/separable_conv2d/ReadVariableOp:value:0*
T0*1
_output_shapes
:         └└*
paddingSAME*
strides
а
+model_1/separable_conv2d_1/separable_conv2dConv2D>model_1/separable_conv2d_1/separable_conv2d/depthwise:output:0Dmodel_1/separable_conv2d_1/separable_conv2d/ReadVariableOp_1:value:0*
T0*1
_output_shapes
:         └└*
paddingVALID*
strides
и
1model_1/separable_conv2d_1/BiasAdd/ReadVariableOpReadVariableOp:model_1_separable_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0┌
"model_1/separable_conv2d_1/BiasAddBiasAdd4model_1/separable_conv2d_1/separable_conv2d:output:09model_1/separable_conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         └└Р
model_1/separable_conv2d_1/ReluRelu+model_1/separable_conv2d_1/BiasAdd:output:0*
T0*1
_output_shapes
:         └└┼
model_1/max_pooling2d/MaxPoolMaxPool-model_1/separable_conv2d_1/Relu:activations:0*1
_output_shapes
:         аа*
ksize
*
paddingSAME*
strides
╞
:model_1/separable_conv2d_2/separable_conv2d/ReadVariableOpReadVariableOpCmodel_1_separable_conv2d_2_separable_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0╩
<model_1/separable_conv2d_2/separable_conv2d/ReadVariableOp_1ReadVariableOpEmodel_1_separable_conv2d_2_separable_conv2d_readvariableop_1_resource*&
_output_shapes
:*
dtype0К
1model_1/separable_conv2d_2/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            К
9model_1/separable_conv2d_2/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      Ю
5model_1/separable_conv2d_2/separable_conv2d/depthwiseDepthwiseConv2dNative&model_1/max_pooling2d/MaxPool:output:0Bmodel_1/separable_conv2d_2/separable_conv2d/ReadVariableOp:value:0*
T0*1
_output_shapes
:         аа*
paddingSAME*
strides
а
+model_1/separable_conv2d_2/separable_conv2dConv2D>model_1/separable_conv2d_2/separable_conv2d/depthwise:output:0Dmodel_1/separable_conv2d_2/separable_conv2d/ReadVariableOp_1:value:0*
T0*1
_output_shapes
:         аа*
paddingVALID*
strides
и
1model_1/separable_conv2d_2/BiasAdd/ReadVariableOpReadVariableOp:model_1_separable_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0┌
"model_1/separable_conv2d_2/BiasAddBiasAdd4model_1/separable_conv2d_2/separable_conv2d:output:09model_1/separable_conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ааР
model_1/separable_conv2d_2/ReluRelu+model_1/separable_conv2d_2/BiasAdd:output:0*
T0*1
_output_shapes
:         аа╞
:model_1/separable_conv2d_3/separable_conv2d/ReadVariableOpReadVariableOpCmodel_1_separable_conv2d_3_separable_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0╩
<model_1/separable_conv2d_3/separable_conv2d/ReadVariableOp_1ReadVariableOpEmodel_1_separable_conv2d_3_separable_conv2d_readvariableop_1_resource*&
_output_shapes
:*
dtype0К
1model_1/separable_conv2d_3/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            К
9model_1/separable_conv2d_3/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      е
5model_1/separable_conv2d_3/separable_conv2d/depthwiseDepthwiseConv2dNative-model_1/separable_conv2d_2/Relu:activations:0Bmodel_1/separable_conv2d_3/separable_conv2d/ReadVariableOp:value:0*
T0*1
_output_shapes
:         аа*
paddingSAME*
strides
а
+model_1/separable_conv2d_3/separable_conv2dConv2D>model_1/separable_conv2d_3/separable_conv2d/depthwise:output:0Dmodel_1/separable_conv2d_3/separable_conv2d/ReadVariableOp_1:value:0*
T0*1
_output_shapes
:         аа*
paddingVALID*
strides
и
1model_1/separable_conv2d_3/BiasAdd/ReadVariableOpReadVariableOp:model_1_separable_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0┌
"model_1/separable_conv2d_3/BiasAddBiasAdd4model_1/separable_conv2d_3/separable_conv2d:output:09model_1/separable_conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ааР
model_1/separable_conv2d_3/ReluRelu+model_1/separable_conv2d_3/BiasAdd:output:0*
T0*1
_output_shapes
:         ааl
model_1/up_sampling2d/ConstConst*
_output_shapes
:*
dtype0*
valueB"а   а   n
model_1/up_sampling2d/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      У
model_1/up_sampling2d/mulMul$model_1/up_sampling2d/Const:output:0&model_1/up_sampling2d/Const_1:output:0*
T0*
_output_shapes
:я
2model_1/up_sampling2d/resize/ResizeNearestNeighborResizeNearestNeighbor-model_1/separable_conv2d_3/Relu:activations:0model_1/up_sampling2d/mul:z:0*
T0*1
_output_shapes
:         └└*
half_pixel_centers(С
model_1/conv2d_transpose/ShapeShapeCmodel_1/up_sampling2d/resize/ResizeNearestNeighbor:resized_images:0*
T0*
_output_shapes
:v
,model_1/conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.model_1/conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.model_1/conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╬
&model_1/conv2d_transpose/strided_sliceStridedSlice'model_1/conv2d_transpose/Shape:output:05model_1/conv2d_transpose/strided_slice/stack:output:07model_1/conv2d_transpose/strided_slice/stack_1:output:07model_1/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
 model_1/conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value
B :└c
 model_1/conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value
B :└b
 model_1/conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B :Ж
model_1/conv2d_transpose/stackPack/model_1/conv2d_transpose/strided_slice:output:0)model_1/conv2d_transpose/stack/1:output:0)model_1/conv2d_transpose/stack/2:output:0)model_1/conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:x
.model_1/conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0model_1/conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0model_1/conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╓
(model_1/conv2d_transpose/strided_slice_1StridedSlice'model_1/conv2d_transpose/stack:output:07model_1/conv2d_transpose/strided_slice_1/stack:output:09model_1/conv2d_transpose/strided_slice_1/stack_1:output:09model_1/conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask┬
8model_1/conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOpAmodel_1_conv2d_transpose_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0╘
)model_1/conv2d_transpose/conv2d_transposeConv2DBackpropInput'model_1/conv2d_transpose/stack:output:0@model_1/conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0Cmodel_1/up_sampling2d/resize/ResizeNearestNeighbor:resized_images:0*
T0*1
_output_shapes
:         └└*
paddingSAME*
strides
д
/model_1/conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp8model_1_conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0╘
 model_1/conv2d_transpose/BiasAddBiasAdd2model_1/conv2d_transpose/conv2d_transpose:output:07model_1/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         └└М
model_1/conv2d_transpose/ReluRelu)model_1/conv2d_transpose/BiasAdd:output:0*
T0*1
_output_shapes
:         └└{
 model_1/conv2d_transpose_1/ShapeShape+model_1/conv2d_transpose/Relu:activations:0*
T0*
_output_shapes
:x
.model_1/conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0model_1/conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0model_1/conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╪
(model_1/conv2d_transpose_1/strided_sliceStridedSlice)model_1/conv2d_transpose_1/Shape:output:07model_1/conv2d_transpose_1/strided_slice/stack:output:09model_1/conv2d_transpose_1/strided_slice/stack_1:output:09model_1/conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
"model_1/conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value
B :└e
"model_1/conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value
B :└d
"model_1/conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B :Р
 model_1/conv2d_transpose_1/stackPack1model_1/conv2d_transpose_1/strided_slice:output:0+model_1/conv2d_transpose_1/stack/1:output:0+model_1/conv2d_transpose_1/stack/2:output:0+model_1/conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:z
0model_1/conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2model_1/conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2model_1/conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:р
*model_1/conv2d_transpose_1/strided_slice_1StridedSlice)model_1/conv2d_transpose_1/stack:output:09model_1/conv2d_transpose_1/strided_slice_1/stack:output:0;model_1/conv2d_transpose_1/strided_slice_1/stack_1:output:0;model_1/conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask╞
:model_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOpCmodel_1_conv2d_transpose_1_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0┬
+model_1/conv2d_transpose_1/conv2d_transposeConv2DBackpropInput)model_1/conv2d_transpose_1/stack:output:0Bmodel_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0+model_1/conv2d_transpose/Relu:activations:0*
T0*1
_output_shapes
:         └└*
paddingSAME*
strides
и
1model_1/conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOp:model_1_conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0┌
"model_1/conv2d_transpose_1/BiasAddBiasAdd4model_1/conv2d_transpose_1/conv2d_transpose:output:09model_1/conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         └└Р
model_1/conv2d_transpose_1/ReluRelu+model_1/conv2d_transpose_1/BiasAdd:output:0*
T0*1
_output_shapes
:         └└Ь
)model_1/my_dense/Tensordot/ReadVariableOpReadVariableOp2model_1_my_dense_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0i
model_1/my_dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:t
model_1/my_dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          }
 model_1/my_dense/Tensordot/ShapeShape-model_1/conv2d_transpose_1/Relu:activations:0*
T0*
_output_shapes
:j
(model_1/my_dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B :  
#model_1/my_dense/Tensordot/GatherV2GatherV2)model_1/my_dense/Tensordot/Shape:output:0(model_1/my_dense/Tensordot/free:output:01model_1/my_dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:l
*model_1/my_dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Г
%model_1/my_dense/Tensordot/GatherV2_1GatherV2)model_1/my_dense/Tensordot/Shape:output:0(model_1/my_dense/Tensordot/axes:output:03model_1/my_dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:j
 model_1/my_dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: б
model_1/my_dense/Tensordot/ProdProd,model_1/my_dense/Tensordot/GatherV2:output:0)model_1/my_dense/Tensordot/Const:output:0*
T0*
_output_shapes
: l
"model_1/my_dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: з
!model_1/my_dense/Tensordot/Prod_1Prod.model_1/my_dense/Tensordot/GatherV2_1:output:0+model_1/my_dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: h
&model_1/my_dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : р
!model_1/my_dense/Tensordot/concatConcatV2(model_1/my_dense/Tensordot/free:output:0(model_1/my_dense/Tensordot/axes:output:0/model_1/my_dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:м
 model_1/my_dense/Tensordot/stackPack(model_1/my_dense/Tensordot/Prod:output:0*model_1/my_dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:╚
$model_1/my_dense/Tensordot/transpose	Transpose-model_1/conv2d_transpose_1/Relu:activations:0*model_1/my_dense/Tensordot/concat:output:0*
T0*1
_output_shapes
:         └└╜
"model_1/my_dense/Tensordot/ReshapeReshape(model_1/my_dense/Tensordot/transpose:y:0)model_1/my_dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  ╜
!model_1/my_dense/Tensordot/MatMulMatMul+model_1/my_dense/Tensordot/Reshape:output:01model_1/my_dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         l
"model_1/my_dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:j
(model_1/my_dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ы
#model_1/my_dense/Tensordot/concat_1ConcatV2,model_1/my_dense/Tensordot/GatherV2:output:0+model_1/my_dense/Tensordot/Const_2:output:01model_1/my_dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:╝
model_1/my_dense/TensordotReshape+model_1/my_dense/Tensordot/MatMul:product:0,model_1/my_dense/Tensordot/concat_1:output:0*
T0*1
_output_shapes
:         └└Ф
'model_1/my_dense/BiasAdd/ReadVariableOpReadVariableOp0model_1_my_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0╡
model_1/my_dense/BiasAddBiasAdd#model_1/my_dense/Tensordot:output:0/model_1/my_dense/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         └└В
model_1/my_dense/SigmoidSigmoid!model_1/my_dense/BiasAdd:output:0*
T0*1
_output_shapes
:         └└a
model_1/reshape/ShapeShapemodel_1/my_dense/Sigmoid:y:0*
T0*
_output_shapes
:m
#model_1/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%model_1/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%model_1/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
model_1/reshape/strided_sliceStridedSlicemodel_1/reshape/Shape:output:0,model_1/reshape/strided_slice/stack:output:0.model_1/reshape/strided_slice/stack_1:output:0.model_1/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskb
model_1/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :└b
model_1/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :└╧
model_1/reshape/Reshape/shapePack&model_1/reshape/strided_slice:output:0(model_1/reshape/Reshape/shape/1:output:0(model_1/reshape/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:а
model_1/reshape/ReshapeReshapemodel_1/my_dense/Sigmoid:y:0&model_1/reshape/Reshape/shape:output:0*
T0*-
_output_shapes
:         └└f
!model_1/tf.math.greater/Greater/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?░
model_1/tf.math.greater/GreaterGreater model_1/reshape/Reshape:output:0*model_1/tf.math.greater/Greater/y:output:0*
T0*-
_output_shapes
:         └└w
2model_1/tf.keras.backend.binary_crossentropy/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Х┐╓3w
2model_1/tf.keras.backend.binary_crossentropy/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╥
0model_1/tf.keras.backend.binary_crossentropy/subSub;model_1/tf.keras.backend.binary_crossentropy/sub/x:output:0;model_1/tf.keras.backend.binary_crossentropy/Const:output:0*
T0*
_output_shapes
: ▌
Bmodel_1/tf.keras.backend.binary_crossentropy/clip_by_value/MinimumMinimum model_1/reshape/Reshape:output:04model_1/tf.keras.backend.binary_crossentropy/sub:z:0*
T0*-
_output_shapes
:         └└В
:model_1/tf.keras.backend.binary_crossentropy/clip_by_valueMaximumFmodel_1/tf.keras.backend.binary_crossentropy/clip_by_value/Minimum:z:0;model_1/tf.keras.backend.binary_crossentropy/Const:output:0*
T0*-
_output_shapes
:         └└w
2model_1/tf.keras.backend.binary_crossentropy/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Х┐╓3ю
0model_1/tf.keras.backend.binary_crossentropy/addAddV2>model_1/tf.keras.backend.binary_crossentropy/clip_by_value:z:0;model_1/tf.keras.backend.binary_crossentropy/add/y:output:0*
T0*-
_output_shapes
:         └└е
0model_1/tf.keras.backend.binary_crossentropy/LogLog4model_1/tf.keras.backend.binary_crossentropy/add:z:0*
T0*-
_output_shapes
:         └└▓
0model_1/tf.keras.backend.binary_crossentropy/mulMultrue_labels4model_1/tf.keras.backend.binary_crossentropy/Log:y:0*
T0*-
_output_shapes
:         └└y
4model_1/tf.keras.backend.binary_crossentropy/sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╜
2model_1/tf.keras.backend.binary_crossentropy/sub_1Sub=model_1/tf.keras.backend.binary_crossentropy/sub_1/x:output:0true_labels*
T0*-
_output_shapes
:         └└y
4model_1/tf.keras.backend.binary_crossentropy/sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Ё
2model_1/tf.keras.backend.binary_crossentropy/sub_2Sub=model_1/tf.keras.backend.binary_crossentropy/sub_2/x:output:0>model_1/tf.keras.backend.binary_crossentropy/clip_by_value:z:0*
T0*-
_output_shapes
:         └└y
4model_1/tf.keras.backend.binary_crossentropy/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *Х┐╓3ъ
2model_1/tf.keras.backend.binary_crossentropy/add_1AddV26model_1/tf.keras.backend.binary_crossentropy/sub_2:z:0=model_1/tf.keras.backend.binary_crossentropy/add_1/y:output:0*
T0*-
_output_shapes
:         └└й
2model_1/tf.keras.backend.binary_crossentropy/Log_1Log6model_1/tf.keras.backend.binary_crossentropy/add_1:z:0*
T0*-
_output_shapes
:         └└с
2model_1/tf.keras.backend.binary_crossentropy/mul_1Mul6model_1/tf.keras.backend.binary_crossentropy/sub_1:z:06model_1/tf.keras.backend.binary_crossentropy/Log_1:y:0*
T0*-
_output_shapes
:         └└с
2model_1/tf.keras.backend.binary_crossentropy/add_2AddV24model_1/tf.keras.backend.binary_crossentropy/mul:z:06model_1/tf.keras.backend.binary_crossentropy/mul_1:z:0*
T0*-
_output_shapes
:         └└з
0model_1/tf.keras.backend.binary_crossentropy/NegNeg6model_1/tf.keras.backend.binary_crossentropy/add_2:z:0*
T0*-
_output_shapes
:         └└К
model_1/tf.cast_1/CastCast#model_1/tf.math.greater/Greater:z:0*

DstT0*

SrcT0
*-
_output_shapes
:         └└u
2model_1/tf.math.reduce_mean/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB w
4model_1/tf.math.reduce_mean/Mean/reduction_indices_1Const*
_output_shapes
: *
dtype0*
valueB ╒
 model_1/tf.math.reduce_mean/MeanMean4model_1/tf.keras.backend.binary_crossentropy/Neg:y:0=model_1/tf.math.reduce_mean/Mean/reduction_indices_1:output:0*
T0*-
_output_shapes
:         └└Е
model_1/tf.math.equal/EqualEqualmodel_1/tf.cast_1/Cast:y:0true_labels*
T0*-
_output_shapes
:         └└Т
model_1/tf.math.multiply/MulMul)model_1/tf.math.reduce_mean/Mean:output:0
weights_in*
T0*-
_output_shapes
:         └└Ж
model_1/tf.cast_2/CastCastmodel_1/tf.math.equal/Equal:z:0*

DstT0*

SrcT0
*-
_output_shapes
:         └└Б
0model_1/tf.math.reduce_sum/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ░
model_1/tf.math.reduce_sum/SumSum model_1/tf.math.multiply/Mul:z:09model_1/tf.math.reduce_sum/Sum/reduction_indices:output:0*
T0*#
_output_shapes
:         Е
model_1/tf.math.multiply_1/MulMulmodel_1/tf.cast_2/Cast:y:0
weights_in*
T0*-
_output_shapes
:         └└Г
2model_1/tf.math.reduce_sum_1/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ╢
 model_1/tf.math.reduce_sum_1/SumSum"model_1/tf.math.multiply_1/Mul:z:0;model_1/tf.math.reduce_sum_1/Sum/reduction_indices:output:0*
T0*#
_output_shapes
:         b
model_1/add_metric/ConstConst*
_output_shapes
:*
dtype0*
valueB: М
model_1/add_metric/SumSum)model_1/tf.math.reduce_sum_1/Sum:output:0!model_1/add_metric/Const:output:0*
T0*
_output_shapes
: ▓
&model_1/add_metric/AssignAddVariableOpAssignAddVariableOp/model_1_add_metric_assignaddvariableop_resourcemodel_1/add_metric/Sum:output:0*
_output_shapes
 *
dtype0k
model_1/add_metric/SizeSize)model_1/tf.math.reduce_sum_1/Sum:output:0*
T0*
_output_shapes
: q
model_1/add_metric/CastCast model_1/add_metric/Size:output:0*

DstT0*

SrcT0*
_output_shapes
: █
(model_1/add_metric/AssignAddVariableOp_1AssignAddVariableOp1model_1_add_metric_assignaddvariableop_1_resourcemodel_1/add_metric/Cast:y:0'^model_1/add_metric/AssignAddVariableOp*
_output_shapes
 *
dtype0ш
,model_1/add_metric/div_no_nan/ReadVariableOpReadVariableOp/model_1_add_metric_assignaddvariableop_resource'^model_1/add_metric/AssignAddVariableOp)^model_1/add_metric/AssignAddVariableOp_1*
_output_shapes
: *
dtype0├
.model_1/add_metric/div_no_nan/ReadVariableOp_1ReadVariableOp1model_1_add_metric_assignaddvariableop_1_resource)^model_1/add_metric/AssignAddVariableOp_1*
_output_shapes
: *
dtype0╕
model_1/add_metric/div_no_nanDivNoNan4model_1/add_metric/div_no_nan/ReadVariableOp:value:06model_1/add_metric/div_no_nan/ReadVariableOp_1:value:0*
T0*
_output_shapes
: k
model_1/add_metric/IdentityIdentity!model_1/add_metric/div_no_nan:z:0*
T0*
_output_shapes
: *(
_construction_contextkEagerRuntime*К
_input_shapesy
w:         └└:         └└:         └└: : : : : : : : : : : : : : : : : : : : 2P
&model_1/add_metric/AssignAddVariableOp&model_1/add_metric/AssignAddVariableOp2T
(model_1/add_metric/AssignAddVariableOp_1(model_1/add_metric/AssignAddVariableOp_12\
,model_1/add_metric/div_no_nan/ReadVariableOp,model_1/add_metric/div_no_nan/ReadVariableOp2`
.model_1/add_metric/div_no_nan/ReadVariableOp_1.model_1/add_metric/div_no_nan/ReadVariableOp_12b
/model_1/conv2d_transpose/BiasAdd/ReadVariableOp/model_1/conv2d_transpose/BiasAdd/ReadVariableOp2t
8model_1/conv2d_transpose/conv2d_transpose/ReadVariableOp8model_1/conv2d_transpose/conv2d_transpose/ReadVariableOp2f
1model_1/conv2d_transpose_1/BiasAdd/ReadVariableOp1model_1/conv2d_transpose_1/BiasAdd/ReadVariableOp2x
:model_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp:model_1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp2R
'model_1/my_dense/BiasAdd/ReadVariableOp'model_1/my_dense/BiasAdd/ReadVariableOp2V
)model_1/my_dense/Tensordot/ReadVariableOp)model_1/my_dense/Tensordot/ReadVariableOp2b
/model_1/separable_conv2d/BiasAdd/ReadVariableOp/model_1/separable_conv2d/BiasAdd/ReadVariableOp2t
8model_1/separable_conv2d/separable_conv2d/ReadVariableOp8model_1/separable_conv2d/separable_conv2d/ReadVariableOp2x
:model_1/separable_conv2d/separable_conv2d/ReadVariableOp_1:model_1/separable_conv2d/separable_conv2d/ReadVariableOp_12f
1model_1/separable_conv2d_1/BiasAdd/ReadVariableOp1model_1/separable_conv2d_1/BiasAdd/ReadVariableOp2x
:model_1/separable_conv2d_1/separable_conv2d/ReadVariableOp:model_1/separable_conv2d_1/separable_conv2d/ReadVariableOp2|
<model_1/separable_conv2d_1/separable_conv2d/ReadVariableOp_1<model_1/separable_conv2d_1/separable_conv2d/ReadVariableOp_12f
1model_1/separable_conv2d_2/BiasAdd/ReadVariableOp1model_1/separable_conv2d_2/BiasAdd/ReadVariableOp2x
:model_1/separable_conv2d_2/separable_conv2d/ReadVariableOp:model_1/separable_conv2d_2/separable_conv2d/ReadVariableOp2|
<model_1/separable_conv2d_2/separable_conv2d/ReadVariableOp_1<model_1/separable_conv2d_2/separable_conv2d/ReadVariableOp_12f
1model_1/separable_conv2d_3/BiasAdd/ReadVariableOp1model_1/separable_conv2d_3/BiasAdd/ReadVariableOp2x
:model_1/separable_conv2d_3/separable_conv2d/ReadVariableOp:model_1/separable_conv2d_3/separable_conv2d/ReadVariableOp2|
<model_1/separable_conv2d_3/separable_conv2d/ReadVariableOp_1<model_1/separable_conv2d_3/separable_conv2d/ReadVariableOp_1:Y U
1
_output_shapes
:         └└
 
_user_specified_nameimg_in:YU
-
_output_shapes
:         └└
$
_user_specified_name
weights_in:ZV
-
_output_shapes
:         └└
%
_user_specified_nametrue_labels
к
Г
L__inference_separable_conv2d_1_layer_call_and_return_conditional_losses_4530

inputsB
(separable_conv2d_readvariableop_resource:D
*separable_conv2d_readvariableop_1_resource:-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвseparable_conv2d/ReadVariableOpв!separable_conv2d/ReadVariableOp_1Р
separable_conv2d/ReadVariableOpReadVariableOp(separable_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ф
!separable_conv2d/ReadVariableOp_1ReadVariableOp*separable_conv2d_readvariableop_1_resource*&
_output_shapes
:*
dtype0o
separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            o
separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      ╪
separable_conv2d/depthwiseDepthwiseConv2dNativeinputs'separable_conv2d/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           *
paddingSAME*
strides
▀
separable_conv2dConv2D#separable_conv2d/depthwise:output:0)separable_conv2d/ReadVariableOp_1:value:0*
T0*A
_output_shapes/
-:+                           *
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0Щ
BiasAddBiasAddseparable_conv2d:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           j
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+                           {
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+                           е
NoOpNoOp^BiasAdd/ReadVariableOp ^separable_conv2d/ReadVariableOp"^separable_conv2d/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:+                           : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
separable_conv2d/ReadVariableOpseparable_conv2d/ReadVariableOp2F
!separable_conv2d/ReadVariableOp_1!separable_conv2d/ReadVariableOp_1:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
╣
▐
&__inference_model_1_layer_call_fn_5517
inputs_0
inputs_1
inputs_2!
unknown:#
	unknown_0:
	unknown_1:#
	unknown_2:#
	unknown_3:
	unknown_4:#
	unknown_5:#
	unknown_6:
	unknown_7:#
	unknown_8:#
	unknown_9:

unknown_10:$

unknown_11:

unknown_12:$

unknown_13:

unknown_14:

unknown_15:

unknown_16:

unknown_17: 

unknown_18: ИвStatefulPartitionedCall█
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1inputs_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18*"
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:         *4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_model_1_layer_call_and_return_conditional_losses_5100*(
_construction_contextkEagerRuntime*К
_input_shapesy
w:         └└:         └└:         └└: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
1
_output_shapes
:         └└
"
_user_specified_name
inputs/0:WS
-
_output_shapes
:         └└
"
_user_specified_name
inputs/1:WS
-
_output_shapes
:         └└
"
_user_specified_name
inputs/2
┌f
е

A__inference_model_1_layer_call_and_return_conditional_losses_5372

img_in

weights_in
true_labels/
separable_conv2d_5285:/
separable_conv2d_5287:#
separable_conv2d_5289:1
separable_conv2d_1_5292:1
separable_conv2d_1_5294:%
separable_conv2d_1_5296:1
separable_conv2d_2_5300:1
separable_conv2d_2_5302:%
separable_conv2d_2_5304:1
separable_conv2d_3_5307:1
separable_conv2d_3_5309:%
separable_conv2d_3_5311:/
conv2d_transpose_5315:#
conv2d_transpose_5317:1
conv2d_transpose_1_5320:%
conv2d_transpose_1_5322:
my_dense_5325:
my_dense_5327:
add_metric_5366: 
add_metric_5368: 
identityИв"add_metric/StatefulPartitionedCallв(conv2d_transpose/StatefulPartitionedCallв*conv2d_transpose_1/StatefulPartitionedCallв my_dense/StatefulPartitionedCallв(separable_conv2d/StatefulPartitionedCallв*separable_conv2d_1/StatefulPartitionedCallв*separable_conv2d_2/StatefulPartitionedCallв*separable_conv2d_3/StatefulPartitionedCallн
(separable_conv2d/StatefulPartitionedCallStatefulPartitionedCallimg_inseparable_conv2d_5285separable_conv2d_5287separable_conv2d_5289*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         └└*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_separable_conv2d_layer_call_and_return_conditional_losses_4501т
*separable_conv2d_1/StatefulPartitionedCallStatefulPartitionedCall1separable_conv2d/StatefulPartitionedCall:output:0separable_conv2d_1_5292separable_conv2d_1_5294separable_conv2d_1_5296*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         └└*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_separable_conv2d_1_layer_call_and_return_conditional_losses_4530ў
max_pooling2d/PartitionedCallPartitionedCall3separable_conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         аа* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_4548╫
*separable_conv2d_2/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0separable_conv2d_2_5300separable_conv2d_2_5302separable_conv2d_2_5304*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         аа*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_separable_conv2d_2_layer_call_and_return_conditional_losses_4571ф
*separable_conv2d_3/StatefulPartitionedCallStatefulPartitionedCall3separable_conv2d_2/StatefulPartitionedCall:output:0separable_conv2d_3_5307separable_conv2d_3_5309separable_conv2d_3_5311*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         аа*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_separable_conv2d_3_layer_call_and_return_conditional_losses_4600З
up_sampling2d/PartitionedCallPartitionedCall3separable_conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_up_sampling2d_layer_call_and_return_conditional_losses_4625─
(conv2d_transpose/StatefulPartitionedCallStatefulPartitionedCall&up_sampling2d/PartitionedCall:output:0conv2d_transpose_5315conv2d_transpose_5317*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_conv2d_transpose_layer_call_and_return_conditional_losses_4666╫
*conv2d_transpose_1/StatefulPartitionedCallStatefulPartitionedCall1conv2d_transpose/StatefulPartitionedCall:output:0conv2d_transpose_1_5320conv2d_transpose_1_5322*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_4711▒
 my_dense/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_1/StatefulPartitionedCall:output:0my_dense_5325my_dense_5327*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_my_dense_layer_call_and_return_conditional_losses_4800▌
reshape/PartitionedCallPartitionedCall)my_dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:         └└* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_reshape_layer_call_and_return_conditional_losses_4819^
tf.math.greater/Greater/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?а
tf.math.greater/GreaterGreater reshape/PartitionedCall:output:0"tf.math.greater/Greater/y:output:0*
T0*-
_output_shapes
:         └└o
*tf.keras.backend.binary_crossentropy/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Х┐╓3o
*tf.keras.backend.binary_crossentropy/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?║
(tf.keras.backend.binary_crossentropy/subSub3tf.keras.backend.binary_crossentropy/sub/x:output:03tf.keras.backend.binary_crossentropy/Const:output:0*
T0*
_output_shapes
: ═
:tf.keras.backend.binary_crossentropy/clip_by_value/MinimumMinimum reshape/PartitionedCall:output:0,tf.keras.backend.binary_crossentropy/sub:z:0*
T0*-
_output_shapes
:         └└ъ
2tf.keras.backend.binary_crossentropy/clip_by_valueMaximum>tf.keras.backend.binary_crossentropy/clip_by_value/Minimum:z:03tf.keras.backend.binary_crossentropy/Const:output:0*
T0*-
_output_shapes
:         └└o
*tf.keras.backend.binary_crossentropy/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Х┐╓3╓
(tf.keras.backend.binary_crossentropy/addAddV26tf.keras.backend.binary_crossentropy/clip_by_value:z:03tf.keras.backend.binary_crossentropy/add/y:output:0*
T0*-
_output_shapes
:         └└Х
(tf.keras.backend.binary_crossentropy/LogLog,tf.keras.backend.binary_crossentropy/add:z:0*
T0*-
_output_shapes
:         └└в
(tf.keras.backend.binary_crossentropy/mulMultrue_labels,tf.keras.backend.binary_crossentropy/Log:y:0*
T0*-
_output_shapes
:         └└q
,tf.keras.backend.binary_crossentropy/sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?н
*tf.keras.backend.binary_crossentropy/sub_1Sub5tf.keras.backend.binary_crossentropy/sub_1/x:output:0true_labels*
T0*-
_output_shapes
:         └└q
,tf.keras.backend.binary_crossentropy/sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╪
*tf.keras.backend.binary_crossentropy/sub_2Sub5tf.keras.backend.binary_crossentropy/sub_2/x:output:06tf.keras.backend.binary_crossentropy/clip_by_value:z:0*
T0*-
_output_shapes
:         └└q
,tf.keras.backend.binary_crossentropy/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *Х┐╓3╥
*tf.keras.backend.binary_crossentropy/add_1AddV2.tf.keras.backend.binary_crossentropy/sub_2:z:05tf.keras.backend.binary_crossentropy/add_1/y:output:0*
T0*-
_output_shapes
:         └└Щ
*tf.keras.backend.binary_crossentropy/Log_1Log.tf.keras.backend.binary_crossentropy/add_1:z:0*
T0*-
_output_shapes
:         └└╔
*tf.keras.backend.binary_crossentropy/mul_1Mul.tf.keras.backend.binary_crossentropy/sub_1:z:0.tf.keras.backend.binary_crossentropy/Log_1:y:0*
T0*-
_output_shapes
:         └└╔
*tf.keras.backend.binary_crossentropy/add_2AddV2,tf.keras.backend.binary_crossentropy/mul:z:0.tf.keras.backend.binary_crossentropy/mul_1:z:0*
T0*-
_output_shapes
:         └└Ч
(tf.keras.backend.binary_crossentropy/NegNeg.tf.keras.backend.binary_crossentropy/add_2:z:0*
T0*-
_output_shapes
:         └└z
tf.cast_1/CastCasttf.math.greater/Greater:z:0*

DstT0*

SrcT0
*-
_output_shapes
:         └└m
*tf.math.reduce_mean/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB o
,tf.math.reduce_mean/Mean/reduction_indices_1Const*
_output_shapes
: *
dtype0*
valueB ╜
tf.math.reduce_mean/MeanMean,tf.keras.backend.binary_crossentropy/Neg:y:05tf.math.reduce_mean/Mean/reduction_indices_1:output:0*
T0*-
_output_shapes
:         └└u
tf.math.equal/EqualEqualtf.cast_1/Cast:y:0true_labels*
T0*-
_output_shapes
:         └└В
tf.math.multiply/MulMul!tf.math.reduce_mean/Mean:output:0
weights_in*
T0*-
_output_shapes
:         └└v
tf.cast_2/CastCasttf.math.equal/Equal:z:0*

DstT0*

SrcT0
*-
_output_shapes
:         └└y
(tf.math.reduce_sum/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      Ш
tf.math.reduce_sum/SumSumtf.math.multiply/Mul:z:01tf.math.reduce_sum/Sum/reduction_indices:output:0*
T0*#
_output_shapes
:         u
tf.math.multiply_1/MulMultf.cast_2/Cast:y:0
weights_in*
T0*-
_output_shapes
:         └└█
add_loss/PartitionedCallPartitionedCalltf.math.reduce_sum/Sum:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:         :         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_add_loss_layer_call_and_return_conditional_losses_4857{
*tf.math.reduce_sum_1/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      Ю
tf.math.reduce_sum_1/SumSumtf.math.multiply_1/Mul:z:03tf.math.reduce_sum_1/Sum/reduction_indices:output:0*
T0*#
_output_shapes
:         Е
"add_metric/StatefulPartitionedCallStatefulPartitionedCall!tf.math.reduce_sum_1/Sum:output:0add_metric_5366add_metric_5368*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_add_metric_layer_call_and_return_conditional_losses_4876l
IdentityIdentity!add_loss/PartitionedCall:output:1^NoOp*
T0*#
_output_shapes
:         Ш
NoOpNoOp#^add_metric/StatefulPartitionedCall)^conv2d_transpose/StatefulPartitionedCall+^conv2d_transpose_1/StatefulPartitionedCall!^my_dense/StatefulPartitionedCall)^separable_conv2d/StatefulPartitionedCall+^separable_conv2d_1/StatefulPartitionedCall+^separable_conv2d_2/StatefulPartitionedCall+^separable_conv2d_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*К
_input_shapesy
w:         └└:         └└:         └└: : : : : : : : : : : : : : : : : : : : 2H
"add_metric/StatefulPartitionedCall"add_metric/StatefulPartitionedCall2T
(conv2d_transpose/StatefulPartitionedCall(conv2d_transpose/StatefulPartitionedCall2X
*conv2d_transpose_1/StatefulPartitionedCall*conv2d_transpose_1/StatefulPartitionedCall2D
 my_dense/StatefulPartitionedCall my_dense/StatefulPartitionedCall2T
(separable_conv2d/StatefulPartitionedCall(separable_conv2d/StatefulPartitionedCall2X
*separable_conv2d_1/StatefulPartitionedCall*separable_conv2d_1/StatefulPartitionedCall2X
*separable_conv2d_2/StatefulPartitionedCall*separable_conv2d_2/StatefulPartitionedCall2X
*separable_conv2d_3/StatefulPartitionedCall*separable_conv2d_3/StatefulPartitionedCall:Y U
1
_output_shapes
:         └└
 
_user_specified_nameimg_in:YU
-
_output_shapes
:         └└
$
_user_specified_name
weights_in:ZV
-
_output_shapes
:         └└
%
_user_specified_nametrue_labels
ў
╦
1__inference_separable_conv2d_2_layer_call_fn_5948

inputs!
unknown:#
	unknown_0:
	unknown_1:
identityИвStatefulPartitionedCallИ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_separable_conv2d_2_layer_call_and_return_conditional_losses_4571Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:+                           : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
є
╔
/__inference_separable_conv2d_layer_call_fn_5884

inputs!
unknown:#
	unknown_0:
	unknown_1:
identityИвStatefulPartitionedCallЖ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_separable_conv2d_layer_call_and_return_conditional_losses_4501Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:+                           : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
├
ж
1__inference_conv2d_transpose_1_layer_call_fn_6060

inputs!
unknown:
	unknown_0:
identityИвStatefulPartitionedCall√
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_4711Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+                           : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
П
c
G__inference_up_sampling2d_layer_call_and_return_conditional_losses_6008

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╜
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:V
ConstConst*
_output_shapes
:*
dtype0*
valueB"      W
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:╡
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4                                    *
half_pixel_centers(Ш
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
к
Г
L__inference_separable_conv2d_3_layer_call_and_return_conditional_losses_5991

inputsB
(separable_conv2d_readvariableop_resource:D
*separable_conv2d_readvariableop_1_resource:-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвseparable_conv2d/ReadVariableOpв!separable_conv2d/ReadVariableOp_1Р
separable_conv2d/ReadVariableOpReadVariableOp(separable_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ф
!separable_conv2d/ReadVariableOp_1ReadVariableOp*separable_conv2d_readvariableop_1_resource*&
_output_shapes
:*
dtype0o
separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            o
separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      ╪
separable_conv2d/depthwiseDepthwiseConv2dNativeinputs'separable_conv2d/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           *
paddingSAME*
strides
▀
separable_conv2dConv2D#separable_conv2d/depthwise:output:0)separable_conv2d/ReadVariableOp_1:value:0*
T0*A
_output_shapes/
-:+                           *
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0Щ
BiasAddBiasAddseparable_conv2d:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           j
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+                           {
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+                           е
NoOpNoOp^BiasAdd/ReadVariableOp ^separable_conv2d/ReadVariableOp"^separable_conv2d/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:+                           : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
separable_conv2d/ReadVariableOpseparable_conv2d/ReadVariableOp2F
!separable_conv2d/ReadVariableOp_1!separable_conv2d/ReadVariableOp_1:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
лю
╛
A__inference_model_1_layer_call_and_return_conditional_losses_5695
inputs_0
inputs_1
inputs_2S
9separable_conv2d_separable_conv2d_readvariableop_resource:U
;separable_conv2d_separable_conv2d_readvariableop_1_resource:>
0separable_conv2d_biasadd_readvariableop_resource:U
;separable_conv2d_1_separable_conv2d_readvariableop_resource:W
=separable_conv2d_1_separable_conv2d_readvariableop_1_resource:@
2separable_conv2d_1_biasadd_readvariableop_resource:U
;separable_conv2d_2_separable_conv2d_readvariableop_resource:W
=separable_conv2d_2_separable_conv2d_readvariableop_1_resource:@
2separable_conv2d_2_biasadd_readvariableop_resource:U
;separable_conv2d_3_separable_conv2d_readvariableop_resource:W
=separable_conv2d_3_separable_conv2d_readvariableop_1_resource:@
2separable_conv2d_3_biasadd_readvariableop_resource:S
9conv2d_transpose_conv2d_transpose_readvariableop_resource:>
0conv2d_transpose_biasadd_readvariableop_resource:U
;conv2d_transpose_1_conv2d_transpose_readvariableop_resource:@
2conv2d_transpose_1_biasadd_readvariableop_resource:<
*my_dense_tensordot_readvariableop_resource:6
(my_dense_biasadd_readvariableop_resource:1
'add_metric_assignaddvariableop_resource: 3
)add_metric_assignaddvariableop_1_resource: 
identityИвadd_metric/AssignAddVariableOpв add_metric/AssignAddVariableOp_1в$add_metric/div_no_nan/ReadVariableOpв&add_metric/div_no_nan/ReadVariableOp_1в'conv2d_transpose/BiasAdd/ReadVariableOpв0conv2d_transpose/conv2d_transpose/ReadVariableOpв)conv2d_transpose_1/BiasAdd/ReadVariableOpв2conv2d_transpose_1/conv2d_transpose/ReadVariableOpвmy_dense/BiasAdd/ReadVariableOpв!my_dense/Tensordot/ReadVariableOpв'separable_conv2d/BiasAdd/ReadVariableOpв0separable_conv2d/separable_conv2d/ReadVariableOpв2separable_conv2d/separable_conv2d/ReadVariableOp_1в)separable_conv2d_1/BiasAdd/ReadVariableOpв2separable_conv2d_1/separable_conv2d/ReadVariableOpв4separable_conv2d_1/separable_conv2d/ReadVariableOp_1в)separable_conv2d_2/BiasAdd/ReadVariableOpв2separable_conv2d_2/separable_conv2d/ReadVariableOpв4separable_conv2d_2/separable_conv2d/ReadVariableOp_1в)separable_conv2d_3/BiasAdd/ReadVariableOpв2separable_conv2d_3/separable_conv2d/ReadVariableOpв4separable_conv2d_3/separable_conv2d/ReadVariableOp_1▓
0separable_conv2d/separable_conv2d/ReadVariableOpReadVariableOp9separable_conv2d_separable_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0╢
2separable_conv2d/separable_conv2d/ReadVariableOp_1ReadVariableOp;separable_conv2d_separable_conv2d_readvariableop_1_resource*&
_output_shapes
:*
dtype0А
'separable_conv2d/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            А
/separable_conv2d/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      ь
+separable_conv2d/separable_conv2d/depthwiseDepthwiseConv2dNativeinputs_08separable_conv2d/separable_conv2d/ReadVariableOp:value:0*
T0*1
_output_shapes
:         └└*
paddingSAME*
strides
В
!separable_conv2d/separable_conv2dConv2D4separable_conv2d/separable_conv2d/depthwise:output:0:separable_conv2d/separable_conv2d/ReadVariableOp_1:value:0*
T0*1
_output_shapes
:         └└*
paddingVALID*
strides
Ф
'separable_conv2d/BiasAdd/ReadVariableOpReadVariableOp0separable_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0╝
separable_conv2d/BiasAddBiasAdd*separable_conv2d/separable_conv2d:output:0/separable_conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         └└|
separable_conv2d/ReluRelu!separable_conv2d/BiasAdd:output:0*
T0*1
_output_shapes
:         └└╢
2separable_conv2d_1/separable_conv2d/ReadVariableOpReadVariableOp;separable_conv2d_1_separable_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0║
4separable_conv2d_1/separable_conv2d/ReadVariableOp_1ReadVariableOp=separable_conv2d_1_separable_conv2d_readvariableop_1_resource*&
_output_shapes
:*
dtype0В
)separable_conv2d_1/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            В
1separable_conv2d_1/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      Л
-separable_conv2d_1/separable_conv2d/depthwiseDepthwiseConv2dNative#separable_conv2d/Relu:activations:0:separable_conv2d_1/separable_conv2d/ReadVariableOp:value:0*
T0*1
_output_shapes
:         └└*
paddingSAME*
strides
И
#separable_conv2d_1/separable_conv2dConv2D6separable_conv2d_1/separable_conv2d/depthwise:output:0<separable_conv2d_1/separable_conv2d/ReadVariableOp_1:value:0*
T0*1
_output_shapes
:         └└*
paddingVALID*
strides
Ш
)separable_conv2d_1/BiasAdd/ReadVariableOpReadVariableOp2separable_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0┬
separable_conv2d_1/BiasAddBiasAdd,separable_conv2d_1/separable_conv2d:output:01separable_conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         └└А
separable_conv2d_1/ReluRelu#separable_conv2d_1/BiasAdd:output:0*
T0*1
_output_shapes
:         └└╡
max_pooling2d/MaxPoolMaxPool%separable_conv2d_1/Relu:activations:0*1
_output_shapes
:         аа*
ksize
*
paddingSAME*
strides
╢
2separable_conv2d_2/separable_conv2d/ReadVariableOpReadVariableOp;separable_conv2d_2_separable_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0║
4separable_conv2d_2/separable_conv2d/ReadVariableOp_1ReadVariableOp=separable_conv2d_2_separable_conv2d_readvariableop_1_resource*&
_output_shapes
:*
dtype0В
)separable_conv2d_2/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            В
1separable_conv2d_2/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      Ж
-separable_conv2d_2/separable_conv2d/depthwiseDepthwiseConv2dNativemax_pooling2d/MaxPool:output:0:separable_conv2d_2/separable_conv2d/ReadVariableOp:value:0*
T0*1
_output_shapes
:         аа*
paddingSAME*
strides
И
#separable_conv2d_2/separable_conv2dConv2D6separable_conv2d_2/separable_conv2d/depthwise:output:0<separable_conv2d_2/separable_conv2d/ReadVariableOp_1:value:0*
T0*1
_output_shapes
:         аа*
paddingVALID*
strides
Ш
)separable_conv2d_2/BiasAdd/ReadVariableOpReadVariableOp2separable_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0┬
separable_conv2d_2/BiasAddBiasAdd,separable_conv2d_2/separable_conv2d:output:01separable_conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ааА
separable_conv2d_2/ReluRelu#separable_conv2d_2/BiasAdd:output:0*
T0*1
_output_shapes
:         аа╢
2separable_conv2d_3/separable_conv2d/ReadVariableOpReadVariableOp;separable_conv2d_3_separable_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0║
4separable_conv2d_3/separable_conv2d/ReadVariableOp_1ReadVariableOp=separable_conv2d_3_separable_conv2d_readvariableop_1_resource*&
_output_shapes
:*
dtype0В
)separable_conv2d_3/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            В
1separable_conv2d_3/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      Н
-separable_conv2d_3/separable_conv2d/depthwiseDepthwiseConv2dNative%separable_conv2d_2/Relu:activations:0:separable_conv2d_3/separable_conv2d/ReadVariableOp:value:0*
T0*1
_output_shapes
:         аа*
paddingSAME*
strides
И
#separable_conv2d_3/separable_conv2dConv2D6separable_conv2d_3/separable_conv2d/depthwise:output:0<separable_conv2d_3/separable_conv2d/ReadVariableOp_1:value:0*
T0*1
_output_shapes
:         аа*
paddingVALID*
strides
Ш
)separable_conv2d_3/BiasAdd/ReadVariableOpReadVariableOp2separable_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0┬
separable_conv2d_3/BiasAddBiasAdd,separable_conv2d_3/separable_conv2d:output:01separable_conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ааА
separable_conv2d_3/ReluRelu#separable_conv2d_3/BiasAdd:output:0*
T0*1
_output_shapes
:         ааd
up_sampling2d/ConstConst*
_output_shapes
:*
dtype0*
valueB"а   а   f
up_sampling2d/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      {
up_sampling2d/mulMulup_sampling2d/Const:output:0up_sampling2d/Const_1:output:0*
T0*
_output_shapes
:╫
*up_sampling2d/resize/ResizeNearestNeighborResizeNearestNeighbor%separable_conv2d_3/Relu:activations:0up_sampling2d/mul:z:0*
T0*1
_output_shapes
:         └└*
half_pixel_centers(Б
conv2d_transpose/ShapeShape;up_sampling2d/resize/ResizeNearestNeighbor:resized_images:0*
T0*
_output_shapes
:n
$conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
conv2d_transpose/strided_sliceStridedSliceconv2d_transpose/Shape:output:0-conv2d_transpose/strided_slice/stack:output:0/conv2d_transpose/strided_slice/stack_1:output:0/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask[
conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value
B :└[
conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value
B :└Z
conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B :▐
conv2d_transpose/stackPack'conv2d_transpose/strided_slice:output:0!conv2d_transpose/stack/1:output:0!conv2d_transpose/stack/2:output:0!conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:p
&conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:о
 conv2d_transpose/strided_slice_1StridedSliceconv2d_transpose/stack:output:0/conv2d_transpose/strided_slice_1/stack:output:01conv2d_transpose/strided_slice_1/stack_1:output:01conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask▓
0conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOp9conv2d_transpose_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0┤
!conv2d_transpose/conv2d_transposeConv2DBackpropInputconv2d_transpose/stack:output:08conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0;up_sampling2d/resize/ResizeNearestNeighbor:resized_images:0*
T0*1
_output_shapes
:         └└*
paddingSAME*
strides
Ф
'conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp0conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0╝
conv2d_transpose/BiasAddBiasAdd*conv2d_transpose/conv2d_transpose:output:0/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         └└|
conv2d_transpose/ReluRelu!conv2d_transpose/BiasAdd:output:0*
T0*1
_output_shapes
:         └└k
conv2d_transpose_1/ShapeShape#conv2d_transpose/Relu:activations:0*
T0*
_output_shapes
:p
&conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:░
 conv2d_transpose_1/strided_sliceStridedSlice!conv2d_transpose_1/Shape:output:0/conv2d_transpose_1/strided_slice/stack:output:01conv2d_transpose_1/strided_slice/stack_1:output:01conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value
B :└]
conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value
B :└\
conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B :ш
conv2d_transpose_1/stackPack)conv2d_transpose_1/strided_slice:output:0#conv2d_transpose_1/stack/1:output:0#conv2d_transpose_1/stack/2:output:0#conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╕
"conv2d_transpose_1/strided_slice_1StridedSlice!conv2d_transpose_1/stack:output:01conv2d_transpose_1/strided_slice_1/stack:output:03conv2d_transpose_1/strided_slice_1/stack_1:output:03conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask╢
2conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_1_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0в
#conv2d_transpose_1/conv2d_transposeConv2DBackpropInput!conv2d_transpose_1/stack:output:0:conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0#conv2d_transpose/Relu:activations:0*
T0*1
_output_shapes
:         └└*
paddingSAME*
strides
Ш
)conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0┬
conv2d_transpose_1/BiasAddBiasAdd,conv2d_transpose_1/conv2d_transpose:output:01conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         └└А
conv2d_transpose_1/ReluRelu#conv2d_transpose_1/BiasAdd:output:0*
T0*1
_output_shapes
:         └└М
!my_dense/Tensordot/ReadVariableOpReadVariableOp*my_dense_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0a
my_dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:l
my_dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          m
my_dense/Tensordot/ShapeShape%conv2d_transpose_1/Relu:activations:0*
T0*
_output_shapes
:b
 my_dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ▀
my_dense/Tensordot/GatherV2GatherV2!my_dense/Tensordot/Shape:output:0 my_dense/Tensordot/free:output:0)my_dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:d
"my_dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : у
my_dense/Tensordot/GatherV2_1GatherV2!my_dense/Tensordot/Shape:output:0 my_dense/Tensordot/axes:output:0+my_dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:b
my_dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: Й
my_dense/Tensordot/ProdProd$my_dense/Tensordot/GatherV2:output:0!my_dense/Tensordot/Const:output:0*
T0*
_output_shapes
: d
my_dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: П
my_dense/Tensordot/Prod_1Prod&my_dense/Tensordot/GatherV2_1:output:0#my_dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: `
my_dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : └
my_dense/Tensordot/concatConcatV2 my_dense/Tensordot/free:output:0 my_dense/Tensordot/axes:output:0'my_dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:Ф
my_dense/Tensordot/stackPack my_dense/Tensordot/Prod:output:0"my_dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:░
my_dense/Tensordot/transpose	Transpose%conv2d_transpose_1/Relu:activations:0"my_dense/Tensordot/concat:output:0*
T0*1
_output_shapes
:         └└е
my_dense/Tensordot/ReshapeReshape my_dense/Tensordot/transpose:y:0!my_dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  е
my_dense/Tensordot/MatMulMatMul#my_dense/Tensordot/Reshape:output:0)my_dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d
my_dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:b
 my_dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ╦
my_dense/Tensordot/concat_1ConcatV2$my_dense/Tensordot/GatherV2:output:0#my_dense/Tensordot/Const_2:output:0)my_dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:д
my_dense/TensordotReshape#my_dense/Tensordot/MatMul:product:0$my_dense/Tensordot/concat_1:output:0*
T0*1
_output_shapes
:         └└Д
my_dense/BiasAdd/ReadVariableOpReadVariableOp(my_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Э
my_dense/BiasAddBiasAddmy_dense/Tensordot:output:0'my_dense/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         └└r
my_dense/SigmoidSigmoidmy_dense/BiasAdd:output:0*
T0*1
_output_shapes
:         └└Q
reshape/ShapeShapemy_dense/Sigmoid:y:0*
T0*
_output_shapes
:e
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:∙
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :└Z
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :└п
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:И
reshape/ReshapeReshapemy_dense/Sigmoid:y:0reshape/Reshape/shape:output:0*
T0*-
_output_shapes
:         └└^
tf.math.greater/Greater/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ш
tf.math.greater/GreaterGreaterreshape/Reshape:output:0"tf.math.greater/Greater/y:output:0*
T0*-
_output_shapes
:         └└o
*tf.keras.backend.binary_crossentropy/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Х┐╓3o
*tf.keras.backend.binary_crossentropy/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?║
(tf.keras.backend.binary_crossentropy/subSub3tf.keras.backend.binary_crossentropy/sub/x:output:03tf.keras.backend.binary_crossentropy/Const:output:0*
T0*
_output_shapes
: ┼
:tf.keras.backend.binary_crossentropy/clip_by_value/MinimumMinimumreshape/Reshape:output:0,tf.keras.backend.binary_crossentropy/sub:z:0*
T0*-
_output_shapes
:         └└ъ
2tf.keras.backend.binary_crossentropy/clip_by_valueMaximum>tf.keras.backend.binary_crossentropy/clip_by_value/Minimum:z:03tf.keras.backend.binary_crossentropy/Const:output:0*
T0*-
_output_shapes
:         └└o
*tf.keras.backend.binary_crossentropy/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Х┐╓3╓
(tf.keras.backend.binary_crossentropy/addAddV26tf.keras.backend.binary_crossentropy/clip_by_value:z:03tf.keras.backend.binary_crossentropy/add/y:output:0*
T0*-
_output_shapes
:         └└Х
(tf.keras.backend.binary_crossentropy/LogLog,tf.keras.backend.binary_crossentropy/add:z:0*
T0*-
_output_shapes
:         └└Я
(tf.keras.backend.binary_crossentropy/mulMulinputs_2,tf.keras.backend.binary_crossentropy/Log:y:0*
T0*-
_output_shapes
:         └└q
,tf.keras.backend.binary_crossentropy/sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?к
*tf.keras.backend.binary_crossentropy/sub_1Sub5tf.keras.backend.binary_crossentropy/sub_1/x:output:0inputs_2*
T0*-
_output_shapes
:         └└q
,tf.keras.backend.binary_crossentropy/sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╪
*tf.keras.backend.binary_crossentropy/sub_2Sub5tf.keras.backend.binary_crossentropy/sub_2/x:output:06tf.keras.backend.binary_crossentropy/clip_by_value:z:0*
T0*-
_output_shapes
:         └└q
,tf.keras.backend.binary_crossentropy/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *Х┐╓3╥
*tf.keras.backend.binary_crossentropy/add_1AddV2.tf.keras.backend.binary_crossentropy/sub_2:z:05tf.keras.backend.binary_crossentropy/add_1/y:output:0*
T0*-
_output_shapes
:         └└Щ
*tf.keras.backend.binary_crossentropy/Log_1Log.tf.keras.backend.binary_crossentropy/add_1:z:0*
T0*-
_output_shapes
:         └└╔
*tf.keras.backend.binary_crossentropy/mul_1Mul.tf.keras.backend.binary_crossentropy/sub_1:z:0.tf.keras.backend.binary_crossentropy/Log_1:y:0*
T0*-
_output_shapes
:         └└╔
*tf.keras.backend.binary_crossentropy/add_2AddV2,tf.keras.backend.binary_crossentropy/mul:z:0.tf.keras.backend.binary_crossentropy/mul_1:z:0*
T0*-
_output_shapes
:         └└Ч
(tf.keras.backend.binary_crossentropy/NegNeg.tf.keras.backend.binary_crossentropy/add_2:z:0*
T0*-
_output_shapes
:         └└z
tf.cast_1/CastCasttf.math.greater/Greater:z:0*

DstT0*

SrcT0
*-
_output_shapes
:         └└m
*tf.math.reduce_mean/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB o
,tf.math.reduce_mean/Mean/reduction_indices_1Const*
_output_shapes
: *
dtype0*
valueB ╜
tf.math.reduce_mean/MeanMean,tf.keras.backend.binary_crossentropy/Neg:y:05tf.math.reduce_mean/Mean/reduction_indices_1:output:0*
T0*-
_output_shapes
:         └└r
tf.math.equal/EqualEqualtf.cast_1/Cast:y:0inputs_2*
T0*-
_output_shapes
:         └└А
tf.math.multiply/MulMul!tf.math.reduce_mean/Mean:output:0inputs_1*
T0*-
_output_shapes
:         └└v
tf.cast_2/CastCasttf.math.equal/Equal:z:0*

DstT0*

SrcT0
*-
_output_shapes
:         └└y
(tf.math.reduce_sum/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      Ш
tf.math.reduce_sum/SumSumtf.math.multiply/Mul:z:01tf.math.reduce_sum/Sum/reduction_indices:output:0*
T0*#
_output_shapes
:         s
tf.math.multiply_1/MulMultf.cast_2/Cast:y:0inputs_1*
T0*-
_output_shapes
:         └└{
*tf.math.reduce_sum_1/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      Ю
tf.math.reduce_sum_1/SumSumtf.math.multiply_1/Mul:z:03tf.math.reduce_sum_1/Sum/reduction_indices:output:0*
T0*#
_output_shapes
:         Z
add_metric/ConstConst*
_output_shapes
:*
dtype0*
valueB: t
add_metric/SumSum!tf.math.reduce_sum_1/Sum:output:0add_metric/Const:output:0*
T0*
_output_shapes
: Ъ
add_metric/AssignAddVariableOpAssignAddVariableOp'add_metric_assignaddvariableop_resourceadd_metric/Sum:output:0*
_output_shapes
 *
dtype0[
add_metric/SizeSize!tf.math.reduce_sum_1/Sum:output:0*
T0*
_output_shapes
: a
add_metric/CastCastadd_metric/Size:output:0*

DstT0*

SrcT0*
_output_shapes
: ╗
 add_metric/AssignAddVariableOp_1AssignAddVariableOp)add_metric_assignaddvariableop_1_resourceadd_metric/Cast:y:0^add_metric/AssignAddVariableOp*
_output_shapes
 *
dtype0╚
$add_metric/div_no_nan/ReadVariableOpReadVariableOp'add_metric_assignaddvariableop_resource^add_metric/AssignAddVariableOp!^add_metric/AssignAddVariableOp_1*
_output_shapes
: *
dtype0л
&add_metric/div_no_nan/ReadVariableOp_1ReadVariableOp)add_metric_assignaddvariableop_1_resource!^add_metric/AssignAddVariableOp_1*
_output_shapes
: *
dtype0а
add_metric/div_no_nanDivNoNan,add_metric/div_no_nan/ReadVariableOp:value:0.add_metric/div_no_nan/ReadVariableOp_1:value:0*
T0*
_output_shapes
: [
add_metric/IdentityIdentityadd_metric/div_no_nan:z:0*
T0*
_output_shapes
: j
IdentityIdentitytf.math.reduce_sum/Sum:output:0^NoOp*
T0*#
_output_shapes
:         ╕
NoOpNoOp^add_metric/AssignAddVariableOp!^add_metric/AssignAddVariableOp_1%^add_metric/div_no_nan/ReadVariableOp'^add_metric/div_no_nan/ReadVariableOp_1(^conv2d_transpose/BiasAdd/ReadVariableOp1^conv2d_transpose/conv2d_transpose/ReadVariableOp*^conv2d_transpose_1/BiasAdd/ReadVariableOp3^conv2d_transpose_1/conv2d_transpose/ReadVariableOp ^my_dense/BiasAdd/ReadVariableOp"^my_dense/Tensordot/ReadVariableOp(^separable_conv2d/BiasAdd/ReadVariableOp1^separable_conv2d/separable_conv2d/ReadVariableOp3^separable_conv2d/separable_conv2d/ReadVariableOp_1*^separable_conv2d_1/BiasAdd/ReadVariableOp3^separable_conv2d_1/separable_conv2d/ReadVariableOp5^separable_conv2d_1/separable_conv2d/ReadVariableOp_1*^separable_conv2d_2/BiasAdd/ReadVariableOp3^separable_conv2d_2/separable_conv2d/ReadVariableOp5^separable_conv2d_2/separable_conv2d/ReadVariableOp_1*^separable_conv2d_3/BiasAdd/ReadVariableOp3^separable_conv2d_3/separable_conv2d/ReadVariableOp5^separable_conv2d_3/separable_conv2d/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*К
_input_shapesy
w:         └└:         └└:         └└: : : : : : : : : : : : : : : : : : : : 2@
add_metric/AssignAddVariableOpadd_metric/AssignAddVariableOp2D
 add_metric/AssignAddVariableOp_1 add_metric/AssignAddVariableOp_12L
$add_metric/div_no_nan/ReadVariableOp$add_metric/div_no_nan/ReadVariableOp2P
&add_metric/div_no_nan/ReadVariableOp_1&add_metric/div_no_nan/ReadVariableOp_12R
'conv2d_transpose/BiasAdd/ReadVariableOp'conv2d_transpose/BiasAdd/ReadVariableOp2d
0conv2d_transpose/conv2d_transpose/ReadVariableOp0conv2d_transpose/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_1/BiasAdd/ReadVariableOp)conv2d_transpose_1/BiasAdd/ReadVariableOp2h
2conv2d_transpose_1/conv2d_transpose/ReadVariableOp2conv2d_transpose_1/conv2d_transpose/ReadVariableOp2B
my_dense/BiasAdd/ReadVariableOpmy_dense/BiasAdd/ReadVariableOp2F
!my_dense/Tensordot/ReadVariableOp!my_dense/Tensordot/ReadVariableOp2R
'separable_conv2d/BiasAdd/ReadVariableOp'separable_conv2d/BiasAdd/ReadVariableOp2d
0separable_conv2d/separable_conv2d/ReadVariableOp0separable_conv2d/separable_conv2d/ReadVariableOp2h
2separable_conv2d/separable_conv2d/ReadVariableOp_12separable_conv2d/separable_conv2d/ReadVariableOp_12V
)separable_conv2d_1/BiasAdd/ReadVariableOp)separable_conv2d_1/BiasAdd/ReadVariableOp2h
2separable_conv2d_1/separable_conv2d/ReadVariableOp2separable_conv2d_1/separable_conv2d/ReadVariableOp2l
4separable_conv2d_1/separable_conv2d/ReadVariableOp_14separable_conv2d_1/separable_conv2d/ReadVariableOp_12V
)separable_conv2d_2/BiasAdd/ReadVariableOp)separable_conv2d_2/BiasAdd/ReadVariableOp2h
2separable_conv2d_2/separable_conv2d/ReadVariableOp2separable_conv2d_2/separable_conv2d/ReadVariableOp2l
4separable_conv2d_2/separable_conv2d/ReadVariableOp_14separable_conv2d_2/separable_conv2d/ReadVariableOp_12V
)separable_conv2d_3/BiasAdd/ReadVariableOp)separable_conv2d_3/BiasAdd/ReadVariableOp2h
2separable_conv2d_3/separable_conv2d/ReadVariableOp2separable_conv2d_3/separable_conv2d/ReadVariableOp2l
4separable_conv2d_3/separable_conv2d/ReadVariableOp_14separable_conv2d_3/separable_conv2d/ReadVariableOp_1:[ W
1
_output_shapes
:         └└
"
_user_specified_name
inputs/0:WS
-
_output_shapes
:         └└
"
_user_specified_name
inputs/1:WS
-
_output_shapes
:         └└
"
_user_specified_name
inputs/2
лю
╛
A__inference_model_1_layer_call_and_return_conditional_losses_5873
inputs_0
inputs_1
inputs_2S
9separable_conv2d_separable_conv2d_readvariableop_resource:U
;separable_conv2d_separable_conv2d_readvariableop_1_resource:>
0separable_conv2d_biasadd_readvariableop_resource:U
;separable_conv2d_1_separable_conv2d_readvariableop_resource:W
=separable_conv2d_1_separable_conv2d_readvariableop_1_resource:@
2separable_conv2d_1_biasadd_readvariableop_resource:U
;separable_conv2d_2_separable_conv2d_readvariableop_resource:W
=separable_conv2d_2_separable_conv2d_readvariableop_1_resource:@
2separable_conv2d_2_biasadd_readvariableop_resource:U
;separable_conv2d_3_separable_conv2d_readvariableop_resource:W
=separable_conv2d_3_separable_conv2d_readvariableop_1_resource:@
2separable_conv2d_3_biasadd_readvariableop_resource:S
9conv2d_transpose_conv2d_transpose_readvariableop_resource:>
0conv2d_transpose_biasadd_readvariableop_resource:U
;conv2d_transpose_1_conv2d_transpose_readvariableop_resource:@
2conv2d_transpose_1_biasadd_readvariableop_resource:<
*my_dense_tensordot_readvariableop_resource:6
(my_dense_biasadd_readvariableop_resource:1
'add_metric_assignaddvariableop_resource: 3
)add_metric_assignaddvariableop_1_resource: 
identityИвadd_metric/AssignAddVariableOpв add_metric/AssignAddVariableOp_1в$add_metric/div_no_nan/ReadVariableOpв&add_metric/div_no_nan/ReadVariableOp_1в'conv2d_transpose/BiasAdd/ReadVariableOpв0conv2d_transpose/conv2d_transpose/ReadVariableOpв)conv2d_transpose_1/BiasAdd/ReadVariableOpв2conv2d_transpose_1/conv2d_transpose/ReadVariableOpвmy_dense/BiasAdd/ReadVariableOpв!my_dense/Tensordot/ReadVariableOpв'separable_conv2d/BiasAdd/ReadVariableOpв0separable_conv2d/separable_conv2d/ReadVariableOpв2separable_conv2d/separable_conv2d/ReadVariableOp_1в)separable_conv2d_1/BiasAdd/ReadVariableOpв2separable_conv2d_1/separable_conv2d/ReadVariableOpв4separable_conv2d_1/separable_conv2d/ReadVariableOp_1в)separable_conv2d_2/BiasAdd/ReadVariableOpв2separable_conv2d_2/separable_conv2d/ReadVariableOpв4separable_conv2d_2/separable_conv2d/ReadVariableOp_1в)separable_conv2d_3/BiasAdd/ReadVariableOpв2separable_conv2d_3/separable_conv2d/ReadVariableOpв4separable_conv2d_3/separable_conv2d/ReadVariableOp_1▓
0separable_conv2d/separable_conv2d/ReadVariableOpReadVariableOp9separable_conv2d_separable_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0╢
2separable_conv2d/separable_conv2d/ReadVariableOp_1ReadVariableOp;separable_conv2d_separable_conv2d_readvariableop_1_resource*&
_output_shapes
:*
dtype0А
'separable_conv2d/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            А
/separable_conv2d/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      ь
+separable_conv2d/separable_conv2d/depthwiseDepthwiseConv2dNativeinputs_08separable_conv2d/separable_conv2d/ReadVariableOp:value:0*
T0*1
_output_shapes
:         └└*
paddingSAME*
strides
В
!separable_conv2d/separable_conv2dConv2D4separable_conv2d/separable_conv2d/depthwise:output:0:separable_conv2d/separable_conv2d/ReadVariableOp_1:value:0*
T0*1
_output_shapes
:         └└*
paddingVALID*
strides
Ф
'separable_conv2d/BiasAdd/ReadVariableOpReadVariableOp0separable_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0╝
separable_conv2d/BiasAddBiasAdd*separable_conv2d/separable_conv2d:output:0/separable_conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         └└|
separable_conv2d/ReluRelu!separable_conv2d/BiasAdd:output:0*
T0*1
_output_shapes
:         └└╢
2separable_conv2d_1/separable_conv2d/ReadVariableOpReadVariableOp;separable_conv2d_1_separable_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0║
4separable_conv2d_1/separable_conv2d/ReadVariableOp_1ReadVariableOp=separable_conv2d_1_separable_conv2d_readvariableop_1_resource*&
_output_shapes
:*
dtype0В
)separable_conv2d_1/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            В
1separable_conv2d_1/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      Л
-separable_conv2d_1/separable_conv2d/depthwiseDepthwiseConv2dNative#separable_conv2d/Relu:activations:0:separable_conv2d_1/separable_conv2d/ReadVariableOp:value:0*
T0*1
_output_shapes
:         └└*
paddingSAME*
strides
И
#separable_conv2d_1/separable_conv2dConv2D6separable_conv2d_1/separable_conv2d/depthwise:output:0<separable_conv2d_1/separable_conv2d/ReadVariableOp_1:value:0*
T0*1
_output_shapes
:         └└*
paddingVALID*
strides
Ш
)separable_conv2d_1/BiasAdd/ReadVariableOpReadVariableOp2separable_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0┬
separable_conv2d_1/BiasAddBiasAdd,separable_conv2d_1/separable_conv2d:output:01separable_conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         └└А
separable_conv2d_1/ReluRelu#separable_conv2d_1/BiasAdd:output:0*
T0*1
_output_shapes
:         └└╡
max_pooling2d/MaxPoolMaxPool%separable_conv2d_1/Relu:activations:0*1
_output_shapes
:         аа*
ksize
*
paddingSAME*
strides
╢
2separable_conv2d_2/separable_conv2d/ReadVariableOpReadVariableOp;separable_conv2d_2_separable_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0║
4separable_conv2d_2/separable_conv2d/ReadVariableOp_1ReadVariableOp=separable_conv2d_2_separable_conv2d_readvariableop_1_resource*&
_output_shapes
:*
dtype0В
)separable_conv2d_2/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            В
1separable_conv2d_2/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      Ж
-separable_conv2d_2/separable_conv2d/depthwiseDepthwiseConv2dNativemax_pooling2d/MaxPool:output:0:separable_conv2d_2/separable_conv2d/ReadVariableOp:value:0*
T0*1
_output_shapes
:         аа*
paddingSAME*
strides
И
#separable_conv2d_2/separable_conv2dConv2D6separable_conv2d_2/separable_conv2d/depthwise:output:0<separable_conv2d_2/separable_conv2d/ReadVariableOp_1:value:0*
T0*1
_output_shapes
:         аа*
paddingVALID*
strides
Ш
)separable_conv2d_2/BiasAdd/ReadVariableOpReadVariableOp2separable_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0┬
separable_conv2d_2/BiasAddBiasAdd,separable_conv2d_2/separable_conv2d:output:01separable_conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ааА
separable_conv2d_2/ReluRelu#separable_conv2d_2/BiasAdd:output:0*
T0*1
_output_shapes
:         аа╢
2separable_conv2d_3/separable_conv2d/ReadVariableOpReadVariableOp;separable_conv2d_3_separable_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0║
4separable_conv2d_3/separable_conv2d/ReadVariableOp_1ReadVariableOp=separable_conv2d_3_separable_conv2d_readvariableop_1_resource*&
_output_shapes
:*
dtype0В
)separable_conv2d_3/separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            В
1separable_conv2d_3/separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      Н
-separable_conv2d_3/separable_conv2d/depthwiseDepthwiseConv2dNative%separable_conv2d_2/Relu:activations:0:separable_conv2d_3/separable_conv2d/ReadVariableOp:value:0*
T0*1
_output_shapes
:         аа*
paddingSAME*
strides
И
#separable_conv2d_3/separable_conv2dConv2D6separable_conv2d_3/separable_conv2d/depthwise:output:0<separable_conv2d_3/separable_conv2d/ReadVariableOp_1:value:0*
T0*1
_output_shapes
:         аа*
paddingVALID*
strides
Ш
)separable_conv2d_3/BiasAdd/ReadVariableOpReadVariableOp2separable_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0┬
separable_conv2d_3/BiasAddBiasAdd,separable_conv2d_3/separable_conv2d:output:01separable_conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ааА
separable_conv2d_3/ReluRelu#separable_conv2d_3/BiasAdd:output:0*
T0*1
_output_shapes
:         ааd
up_sampling2d/ConstConst*
_output_shapes
:*
dtype0*
valueB"а   а   f
up_sampling2d/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      {
up_sampling2d/mulMulup_sampling2d/Const:output:0up_sampling2d/Const_1:output:0*
T0*
_output_shapes
:╫
*up_sampling2d/resize/ResizeNearestNeighborResizeNearestNeighbor%separable_conv2d_3/Relu:activations:0up_sampling2d/mul:z:0*
T0*1
_output_shapes
:         └└*
half_pixel_centers(Б
conv2d_transpose/ShapeShape;up_sampling2d/resize/ResizeNearestNeighbor:resized_images:0*
T0*
_output_shapes
:n
$conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
conv2d_transpose/strided_sliceStridedSliceconv2d_transpose/Shape:output:0-conv2d_transpose/strided_slice/stack:output:0/conv2d_transpose/strided_slice/stack_1:output:0/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask[
conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value
B :└[
conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value
B :└Z
conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B :▐
conv2d_transpose/stackPack'conv2d_transpose/strided_slice:output:0!conv2d_transpose/stack/1:output:0!conv2d_transpose/stack/2:output:0!conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:p
&conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:о
 conv2d_transpose/strided_slice_1StridedSliceconv2d_transpose/stack:output:0/conv2d_transpose/strided_slice_1/stack:output:01conv2d_transpose/strided_slice_1/stack_1:output:01conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask▓
0conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOp9conv2d_transpose_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0┤
!conv2d_transpose/conv2d_transposeConv2DBackpropInputconv2d_transpose/stack:output:08conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0;up_sampling2d/resize/ResizeNearestNeighbor:resized_images:0*
T0*1
_output_shapes
:         └└*
paddingSAME*
strides
Ф
'conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp0conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0╝
conv2d_transpose/BiasAddBiasAdd*conv2d_transpose/conv2d_transpose:output:0/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         └└|
conv2d_transpose/ReluRelu!conv2d_transpose/BiasAdd:output:0*
T0*1
_output_shapes
:         └└k
conv2d_transpose_1/ShapeShape#conv2d_transpose/Relu:activations:0*
T0*
_output_shapes
:p
&conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:░
 conv2d_transpose_1/strided_sliceStridedSlice!conv2d_transpose_1/Shape:output:0/conv2d_transpose_1/strided_slice/stack:output:01conv2d_transpose_1/strided_slice/stack_1:output:01conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value
B :└]
conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value
B :└\
conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B :ш
conv2d_transpose_1/stackPack)conv2d_transpose_1/strided_slice:output:0#conv2d_transpose_1/stack/1:output:0#conv2d_transpose_1/stack/2:output:0#conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╕
"conv2d_transpose_1/strided_slice_1StridedSlice!conv2d_transpose_1/stack:output:01conv2d_transpose_1/strided_slice_1/stack:output:03conv2d_transpose_1/strided_slice_1/stack_1:output:03conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask╢
2conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_1_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0в
#conv2d_transpose_1/conv2d_transposeConv2DBackpropInput!conv2d_transpose_1/stack:output:0:conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0#conv2d_transpose/Relu:activations:0*
T0*1
_output_shapes
:         └└*
paddingSAME*
strides
Ш
)conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0┬
conv2d_transpose_1/BiasAddBiasAdd,conv2d_transpose_1/conv2d_transpose:output:01conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         └└А
conv2d_transpose_1/ReluRelu#conv2d_transpose_1/BiasAdd:output:0*
T0*1
_output_shapes
:         └└М
!my_dense/Tensordot/ReadVariableOpReadVariableOp*my_dense_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0a
my_dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:l
my_dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          m
my_dense/Tensordot/ShapeShape%conv2d_transpose_1/Relu:activations:0*
T0*
_output_shapes
:b
 my_dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ▀
my_dense/Tensordot/GatherV2GatherV2!my_dense/Tensordot/Shape:output:0 my_dense/Tensordot/free:output:0)my_dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:d
"my_dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : у
my_dense/Tensordot/GatherV2_1GatherV2!my_dense/Tensordot/Shape:output:0 my_dense/Tensordot/axes:output:0+my_dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:b
my_dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: Й
my_dense/Tensordot/ProdProd$my_dense/Tensordot/GatherV2:output:0!my_dense/Tensordot/Const:output:0*
T0*
_output_shapes
: d
my_dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: П
my_dense/Tensordot/Prod_1Prod&my_dense/Tensordot/GatherV2_1:output:0#my_dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: `
my_dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : └
my_dense/Tensordot/concatConcatV2 my_dense/Tensordot/free:output:0 my_dense/Tensordot/axes:output:0'my_dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:Ф
my_dense/Tensordot/stackPack my_dense/Tensordot/Prod:output:0"my_dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:░
my_dense/Tensordot/transpose	Transpose%conv2d_transpose_1/Relu:activations:0"my_dense/Tensordot/concat:output:0*
T0*1
_output_shapes
:         └└е
my_dense/Tensordot/ReshapeReshape my_dense/Tensordot/transpose:y:0!my_dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  е
my_dense/Tensordot/MatMulMatMul#my_dense/Tensordot/Reshape:output:0)my_dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d
my_dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:b
 my_dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ╦
my_dense/Tensordot/concat_1ConcatV2$my_dense/Tensordot/GatherV2:output:0#my_dense/Tensordot/Const_2:output:0)my_dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:д
my_dense/TensordotReshape#my_dense/Tensordot/MatMul:product:0$my_dense/Tensordot/concat_1:output:0*
T0*1
_output_shapes
:         └└Д
my_dense/BiasAdd/ReadVariableOpReadVariableOp(my_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Э
my_dense/BiasAddBiasAddmy_dense/Tensordot:output:0'my_dense/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         └└r
my_dense/SigmoidSigmoidmy_dense/BiasAdd:output:0*
T0*1
_output_shapes
:         └└Q
reshape/ShapeShapemy_dense/Sigmoid:y:0*
T0*
_output_shapes
:e
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:∙
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :└Z
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :└п
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:И
reshape/ReshapeReshapemy_dense/Sigmoid:y:0reshape/Reshape/shape:output:0*
T0*-
_output_shapes
:         └└^
tf.math.greater/Greater/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ш
tf.math.greater/GreaterGreaterreshape/Reshape:output:0"tf.math.greater/Greater/y:output:0*
T0*-
_output_shapes
:         └└o
*tf.keras.backend.binary_crossentropy/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Х┐╓3o
*tf.keras.backend.binary_crossentropy/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?║
(tf.keras.backend.binary_crossentropy/subSub3tf.keras.backend.binary_crossentropy/sub/x:output:03tf.keras.backend.binary_crossentropy/Const:output:0*
T0*
_output_shapes
: ┼
:tf.keras.backend.binary_crossentropy/clip_by_value/MinimumMinimumreshape/Reshape:output:0,tf.keras.backend.binary_crossentropy/sub:z:0*
T0*-
_output_shapes
:         └└ъ
2tf.keras.backend.binary_crossentropy/clip_by_valueMaximum>tf.keras.backend.binary_crossentropy/clip_by_value/Minimum:z:03tf.keras.backend.binary_crossentropy/Const:output:0*
T0*-
_output_shapes
:         └└o
*tf.keras.backend.binary_crossentropy/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Х┐╓3╓
(tf.keras.backend.binary_crossentropy/addAddV26tf.keras.backend.binary_crossentropy/clip_by_value:z:03tf.keras.backend.binary_crossentropy/add/y:output:0*
T0*-
_output_shapes
:         └└Х
(tf.keras.backend.binary_crossentropy/LogLog,tf.keras.backend.binary_crossentropy/add:z:0*
T0*-
_output_shapes
:         └└Я
(tf.keras.backend.binary_crossentropy/mulMulinputs_2,tf.keras.backend.binary_crossentropy/Log:y:0*
T0*-
_output_shapes
:         └└q
,tf.keras.backend.binary_crossentropy/sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?к
*tf.keras.backend.binary_crossentropy/sub_1Sub5tf.keras.backend.binary_crossentropy/sub_1/x:output:0inputs_2*
T0*-
_output_shapes
:         └└q
,tf.keras.backend.binary_crossentropy/sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╪
*tf.keras.backend.binary_crossentropy/sub_2Sub5tf.keras.backend.binary_crossentropy/sub_2/x:output:06tf.keras.backend.binary_crossentropy/clip_by_value:z:0*
T0*-
_output_shapes
:         └└q
,tf.keras.backend.binary_crossentropy/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *Х┐╓3╥
*tf.keras.backend.binary_crossentropy/add_1AddV2.tf.keras.backend.binary_crossentropy/sub_2:z:05tf.keras.backend.binary_crossentropy/add_1/y:output:0*
T0*-
_output_shapes
:         └└Щ
*tf.keras.backend.binary_crossentropy/Log_1Log.tf.keras.backend.binary_crossentropy/add_1:z:0*
T0*-
_output_shapes
:         └└╔
*tf.keras.backend.binary_crossentropy/mul_1Mul.tf.keras.backend.binary_crossentropy/sub_1:z:0.tf.keras.backend.binary_crossentropy/Log_1:y:0*
T0*-
_output_shapes
:         └└╔
*tf.keras.backend.binary_crossentropy/add_2AddV2,tf.keras.backend.binary_crossentropy/mul:z:0.tf.keras.backend.binary_crossentropy/mul_1:z:0*
T0*-
_output_shapes
:         └└Ч
(tf.keras.backend.binary_crossentropy/NegNeg.tf.keras.backend.binary_crossentropy/add_2:z:0*
T0*-
_output_shapes
:         └└z
tf.cast_1/CastCasttf.math.greater/Greater:z:0*

DstT0*

SrcT0
*-
_output_shapes
:         └└m
*tf.math.reduce_mean/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB o
,tf.math.reduce_mean/Mean/reduction_indices_1Const*
_output_shapes
: *
dtype0*
valueB ╜
tf.math.reduce_mean/MeanMean,tf.keras.backend.binary_crossentropy/Neg:y:05tf.math.reduce_mean/Mean/reduction_indices_1:output:0*
T0*-
_output_shapes
:         └└r
tf.math.equal/EqualEqualtf.cast_1/Cast:y:0inputs_2*
T0*-
_output_shapes
:         └└А
tf.math.multiply/MulMul!tf.math.reduce_mean/Mean:output:0inputs_1*
T0*-
_output_shapes
:         └└v
tf.cast_2/CastCasttf.math.equal/Equal:z:0*

DstT0*

SrcT0
*-
_output_shapes
:         └└y
(tf.math.reduce_sum/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      Ш
tf.math.reduce_sum/SumSumtf.math.multiply/Mul:z:01tf.math.reduce_sum/Sum/reduction_indices:output:0*
T0*#
_output_shapes
:         s
tf.math.multiply_1/MulMultf.cast_2/Cast:y:0inputs_1*
T0*-
_output_shapes
:         └└{
*tf.math.reduce_sum_1/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      Ю
tf.math.reduce_sum_1/SumSumtf.math.multiply_1/Mul:z:03tf.math.reduce_sum_1/Sum/reduction_indices:output:0*
T0*#
_output_shapes
:         Z
add_metric/ConstConst*
_output_shapes
:*
dtype0*
valueB: t
add_metric/SumSum!tf.math.reduce_sum_1/Sum:output:0add_metric/Const:output:0*
T0*
_output_shapes
: Ъ
add_metric/AssignAddVariableOpAssignAddVariableOp'add_metric_assignaddvariableop_resourceadd_metric/Sum:output:0*
_output_shapes
 *
dtype0[
add_metric/SizeSize!tf.math.reduce_sum_1/Sum:output:0*
T0*
_output_shapes
: a
add_metric/CastCastadd_metric/Size:output:0*

DstT0*

SrcT0*
_output_shapes
: ╗
 add_metric/AssignAddVariableOp_1AssignAddVariableOp)add_metric_assignaddvariableop_1_resourceadd_metric/Cast:y:0^add_metric/AssignAddVariableOp*
_output_shapes
 *
dtype0╚
$add_metric/div_no_nan/ReadVariableOpReadVariableOp'add_metric_assignaddvariableop_resource^add_metric/AssignAddVariableOp!^add_metric/AssignAddVariableOp_1*
_output_shapes
: *
dtype0л
&add_metric/div_no_nan/ReadVariableOp_1ReadVariableOp)add_metric_assignaddvariableop_1_resource!^add_metric/AssignAddVariableOp_1*
_output_shapes
: *
dtype0а
add_metric/div_no_nanDivNoNan,add_metric/div_no_nan/ReadVariableOp:value:0.add_metric/div_no_nan/ReadVariableOp_1:value:0*
T0*
_output_shapes
: [
add_metric/IdentityIdentityadd_metric/div_no_nan:z:0*
T0*
_output_shapes
: j
IdentityIdentitytf.math.reduce_sum/Sum:output:0^NoOp*
T0*#
_output_shapes
:         ╕
NoOpNoOp^add_metric/AssignAddVariableOp!^add_metric/AssignAddVariableOp_1%^add_metric/div_no_nan/ReadVariableOp'^add_metric/div_no_nan/ReadVariableOp_1(^conv2d_transpose/BiasAdd/ReadVariableOp1^conv2d_transpose/conv2d_transpose/ReadVariableOp*^conv2d_transpose_1/BiasAdd/ReadVariableOp3^conv2d_transpose_1/conv2d_transpose/ReadVariableOp ^my_dense/BiasAdd/ReadVariableOp"^my_dense/Tensordot/ReadVariableOp(^separable_conv2d/BiasAdd/ReadVariableOp1^separable_conv2d/separable_conv2d/ReadVariableOp3^separable_conv2d/separable_conv2d/ReadVariableOp_1*^separable_conv2d_1/BiasAdd/ReadVariableOp3^separable_conv2d_1/separable_conv2d/ReadVariableOp5^separable_conv2d_1/separable_conv2d/ReadVariableOp_1*^separable_conv2d_2/BiasAdd/ReadVariableOp3^separable_conv2d_2/separable_conv2d/ReadVariableOp5^separable_conv2d_2/separable_conv2d/ReadVariableOp_1*^separable_conv2d_3/BiasAdd/ReadVariableOp3^separable_conv2d_3/separable_conv2d/ReadVariableOp5^separable_conv2d_3/separable_conv2d/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*К
_input_shapesy
w:         └└:         └└:         └└: : : : : : : : : : : : : : : : : : : : 2@
add_metric/AssignAddVariableOpadd_metric/AssignAddVariableOp2D
 add_metric/AssignAddVariableOp_1 add_metric/AssignAddVariableOp_12L
$add_metric/div_no_nan/ReadVariableOp$add_metric/div_no_nan/ReadVariableOp2P
&add_metric/div_no_nan/ReadVariableOp_1&add_metric/div_no_nan/ReadVariableOp_12R
'conv2d_transpose/BiasAdd/ReadVariableOp'conv2d_transpose/BiasAdd/ReadVariableOp2d
0conv2d_transpose/conv2d_transpose/ReadVariableOp0conv2d_transpose/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_1/BiasAdd/ReadVariableOp)conv2d_transpose_1/BiasAdd/ReadVariableOp2h
2conv2d_transpose_1/conv2d_transpose/ReadVariableOp2conv2d_transpose_1/conv2d_transpose/ReadVariableOp2B
my_dense/BiasAdd/ReadVariableOpmy_dense/BiasAdd/ReadVariableOp2F
!my_dense/Tensordot/ReadVariableOp!my_dense/Tensordot/ReadVariableOp2R
'separable_conv2d/BiasAdd/ReadVariableOp'separable_conv2d/BiasAdd/ReadVariableOp2d
0separable_conv2d/separable_conv2d/ReadVariableOp0separable_conv2d/separable_conv2d/ReadVariableOp2h
2separable_conv2d/separable_conv2d/ReadVariableOp_12separable_conv2d/separable_conv2d/ReadVariableOp_12V
)separable_conv2d_1/BiasAdd/ReadVariableOp)separable_conv2d_1/BiasAdd/ReadVariableOp2h
2separable_conv2d_1/separable_conv2d/ReadVariableOp2separable_conv2d_1/separable_conv2d/ReadVariableOp2l
4separable_conv2d_1/separable_conv2d/ReadVariableOp_14separable_conv2d_1/separable_conv2d/ReadVariableOp_12V
)separable_conv2d_2/BiasAdd/ReadVariableOp)separable_conv2d_2/BiasAdd/ReadVariableOp2h
2separable_conv2d_2/separable_conv2d/ReadVariableOp2separable_conv2d_2/separable_conv2d/ReadVariableOp2l
4separable_conv2d_2/separable_conv2d/ReadVariableOp_14separable_conv2d_2/separable_conv2d/ReadVariableOp_12V
)separable_conv2d_3/BiasAdd/ReadVariableOp)separable_conv2d_3/BiasAdd/ReadVariableOp2h
2separable_conv2d_3/separable_conv2d/ReadVariableOp2separable_conv2d_3/separable_conv2d/ReadVariableOp2l
4separable_conv2d_3/separable_conv2d/ReadVariableOp_14separable_conv2d_3/separable_conv2d/ReadVariableOp_1:[ W
1
_output_shapes
:         └└
"
_user_specified_name
inputs/0:WS
-
_output_shapes
:         └└
"
_user_specified_name
inputs/1:WS
-
_output_shapes
:         └└
"
_user_specified_name
inputs/2
П
c
G__inference_up_sampling2d_layer_call_and_return_conditional_losses_4625

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╜
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:V
ConstConst*
_output_shapes
:*
dtype0*
valueB"      W
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:╡
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4                                    *
half_pixel_centers(Ш
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
└!
Ч
J__inference_conv2d_transpose_layer_call_and_return_conditional_losses_4666

inputsB
(conv2d_transpose_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвconv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┘
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┘
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┘
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskР
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0▄
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+                           *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0Щ
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           j
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+                           {
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+                           Б
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+                           : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
к
Г
L__inference_separable_conv2d_2_layer_call_and_return_conditional_losses_4571

inputsB
(separable_conv2d_readvariableop_resource:D
*separable_conv2d_readvariableop_1_resource:-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвseparable_conv2d/ReadVariableOpв!separable_conv2d/ReadVariableOp_1Р
separable_conv2d/ReadVariableOpReadVariableOp(separable_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ф
!separable_conv2d/ReadVariableOp_1ReadVariableOp*separable_conv2d_readvariableop_1_resource*&
_output_shapes
:*
dtype0o
separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            o
separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      ╪
separable_conv2d/depthwiseDepthwiseConv2dNativeinputs'separable_conv2d/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           *
paddingSAME*
strides
▀
separable_conv2dConv2D#separable_conv2d/depthwise:output:0)separable_conv2d/ReadVariableOp_1:value:0*
T0*A
_output_shapes/
-:+                           *
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0Щ
BiasAddBiasAddseparable_conv2d:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           j
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+                           {
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+                           е
NoOpNoOp^BiasAdd/ReadVariableOp ^separable_conv2d/ReadVariableOp"^separable_conv2d/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:+                           : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
separable_conv2d/ReadVariableOpseparable_conv2d/ReadVariableOp2F
!separable_conv2d/ReadVariableOp_1!separable_conv2d/ReadVariableOp_1:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
╒
Я
D__inference_add_metric_layer_call_and_return_conditional_losses_4876

inputs&
assignaddvariableop_resource: (
assignaddvariableop_1_resource: 

identity_1ИвAssignAddVariableOpвAssignAddVariableOp_1вdiv_no_nan/ReadVariableOpвdiv_no_nan/ReadVariableOp_1O
ConstConst*
_output_shapes
:*
dtype0*
valueB: C
SumSuminputsConst:output:0*
T0*
_output_shapes
: y
AssignAddVariableOpAssignAddVariableOpassignaddvariableop_resourceSum:output:0*
_output_shapes
 *
dtype05
SizeSizeinputs*
T0*
_output_shapes
: K
CastCastSize:output:0*

DstT0*

SrcT0*
_output_shapes
: П
AssignAddVariableOp_1AssignAddVariableOpassignaddvariableop_1_resourceCast:y:0^AssignAddVariableOp*
_output_shapes
 *
dtype0Ь
div_no_nan/ReadVariableOpReadVariableOpassignaddvariableop_resource^AssignAddVariableOp^AssignAddVariableOp_1*
_output_shapes
: *
dtype0К
div_no_nan/ReadVariableOp_1ReadVariableOpassignaddvariableop_1_resource^AssignAddVariableOp_1*
_output_shapes
: *
dtype0

div_no_nanDivNoNan!div_no_nan/ReadVariableOp:value:0#div_no_nan/ReadVariableOp_1:value:0*
T0*
_output_shapes
: E
IdentityIdentitydiv_no_nan:z:0*
T0*
_output_shapes
: S

Identity_1Identityinputs^NoOp*
T0*#
_output_shapes
:         о
NoOpNoOp^AssignAddVariableOp^AssignAddVariableOp_1^div_no_nan/ReadVariableOp^div_no_nan/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         : : 2*
AssignAddVariableOpAssignAddVariableOp2.
AssignAddVariableOp_1AssignAddVariableOp_126
div_no_nan/ReadVariableOpdiv_no_nan/ReadVariableOp2:
div_no_nan/ReadVariableOp_1div_no_nan/ReadVariableOp_1:K G
#
_output_shapes
:         
 
_user_specified_nameinputs
└
∙
B__inference_my_dense_layer_call_and_return_conditional_losses_4800

inputs3
!tensordot_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвTensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:c
Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ╗
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ┐
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ь
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:П
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*A
_output_shapes/
-:+                           К
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:                  К
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         [
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : з
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Щ
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*A
_output_shapes/
-:+                           r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0Т
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           p
SigmoidSigmoidBiasAdd:output:0*
T0*A
_output_shapes/
-:+                           t
IdentityIdentitySigmoid:y:0^NoOp*
T0*A
_output_shapes/
-:+                           z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+                           : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
┐f
а

A__inference_model_1_layer_call_and_return_conditional_losses_5100

inputs
inputs_1
inputs_2/
separable_conv2d_5013:/
separable_conv2d_5015:#
separable_conv2d_5017:1
separable_conv2d_1_5020:1
separable_conv2d_1_5022:%
separable_conv2d_1_5024:1
separable_conv2d_2_5028:1
separable_conv2d_2_5030:%
separable_conv2d_2_5032:1
separable_conv2d_3_5035:1
separable_conv2d_3_5037:%
separable_conv2d_3_5039:/
conv2d_transpose_5043:#
conv2d_transpose_5045:1
conv2d_transpose_1_5048:%
conv2d_transpose_1_5050:
my_dense_5053:
my_dense_5055:
add_metric_5094: 
add_metric_5096: 
identityИв"add_metric/StatefulPartitionedCallв(conv2d_transpose/StatefulPartitionedCallв*conv2d_transpose_1/StatefulPartitionedCallв my_dense/StatefulPartitionedCallв(separable_conv2d/StatefulPartitionedCallв*separable_conv2d_1/StatefulPartitionedCallв*separable_conv2d_2/StatefulPartitionedCallв*separable_conv2d_3/StatefulPartitionedCallн
(separable_conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsseparable_conv2d_5013separable_conv2d_5015separable_conv2d_5017*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         └└*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_separable_conv2d_layer_call_and_return_conditional_losses_4501т
*separable_conv2d_1/StatefulPartitionedCallStatefulPartitionedCall1separable_conv2d/StatefulPartitionedCall:output:0separable_conv2d_1_5020separable_conv2d_1_5022separable_conv2d_1_5024*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         └└*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_separable_conv2d_1_layer_call_and_return_conditional_losses_4530ў
max_pooling2d/PartitionedCallPartitionedCall3separable_conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         аа* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_4548╫
*separable_conv2d_2/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0separable_conv2d_2_5028separable_conv2d_2_5030separable_conv2d_2_5032*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         аа*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_separable_conv2d_2_layer_call_and_return_conditional_losses_4571ф
*separable_conv2d_3/StatefulPartitionedCallStatefulPartitionedCall3separable_conv2d_2/StatefulPartitionedCall:output:0separable_conv2d_3_5035separable_conv2d_3_5037separable_conv2d_3_5039*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         аа*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_separable_conv2d_3_layer_call_and_return_conditional_losses_4600З
up_sampling2d/PartitionedCallPartitionedCall3separable_conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_up_sampling2d_layer_call_and_return_conditional_losses_4625─
(conv2d_transpose/StatefulPartitionedCallStatefulPartitionedCall&up_sampling2d/PartitionedCall:output:0conv2d_transpose_5043conv2d_transpose_5045*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_conv2d_transpose_layer_call_and_return_conditional_losses_4666╫
*conv2d_transpose_1/StatefulPartitionedCallStatefulPartitionedCall1conv2d_transpose/StatefulPartitionedCall:output:0conv2d_transpose_1_5048conv2d_transpose_1_5050*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_4711▒
 my_dense/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_1/StatefulPartitionedCall:output:0my_dense_5053my_dense_5055*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_my_dense_layer_call_and_return_conditional_losses_4800▌
reshape/PartitionedCallPartitionedCall)my_dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:         └└* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_reshape_layer_call_and_return_conditional_losses_4819^
tf.math.greater/Greater/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?а
tf.math.greater/GreaterGreater reshape/PartitionedCall:output:0"tf.math.greater/Greater/y:output:0*
T0*-
_output_shapes
:         └└o
*tf.keras.backend.binary_crossentropy/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Х┐╓3o
*tf.keras.backend.binary_crossentropy/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?║
(tf.keras.backend.binary_crossentropy/subSub3tf.keras.backend.binary_crossentropy/sub/x:output:03tf.keras.backend.binary_crossentropy/Const:output:0*
T0*
_output_shapes
: ═
:tf.keras.backend.binary_crossentropy/clip_by_value/MinimumMinimum reshape/PartitionedCall:output:0,tf.keras.backend.binary_crossentropy/sub:z:0*
T0*-
_output_shapes
:         └└ъ
2tf.keras.backend.binary_crossentropy/clip_by_valueMaximum>tf.keras.backend.binary_crossentropy/clip_by_value/Minimum:z:03tf.keras.backend.binary_crossentropy/Const:output:0*
T0*-
_output_shapes
:         └└o
*tf.keras.backend.binary_crossentropy/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Х┐╓3╓
(tf.keras.backend.binary_crossentropy/addAddV26tf.keras.backend.binary_crossentropy/clip_by_value:z:03tf.keras.backend.binary_crossentropy/add/y:output:0*
T0*-
_output_shapes
:         └└Х
(tf.keras.backend.binary_crossentropy/LogLog,tf.keras.backend.binary_crossentropy/add:z:0*
T0*-
_output_shapes
:         └└Я
(tf.keras.backend.binary_crossentropy/mulMulinputs_2,tf.keras.backend.binary_crossentropy/Log:y:0*
T0*-
_output_shapes
:         └└q
,tf.keras.backend.binary_crossentropy/sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?к
*tf.keras.backend.binary_crossentropy/sub_1Sub5tf.keras.backend.binary_crossentropy/sub_1/x:output:0inputs_2*
T0*-
_output_shapes
:         └└q
,tf.keras.backend.binary_crossentropy/sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╪
*tf.keras.backend.binary_crossentropy/sub_2Sub5tf.keras.backend.binary_crossentropy/sub_2/x:output:06tf.keras.backend.binary_crossentropy/clip_by_value:z:0*
T0*-
_output_shapes
:         └└q
,tf.keras.backend.binary_crossentropy/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *Х┐╓3╥
*tf.keras.backend.binary_crossentropy/add_1AddV2.tf.keras.backend.binary_crossentropy/sub_2:z:05tf.keras.backend.binary_crossentropy/add_1/y:output:0*
T0*-
_output_shapes
:         └└Щ
*tf.keras.backend.binary_crossentropy/Log_1Log.tf.keras.backend.binary_crossentropy/add_1:z:0*
T0*-
_output_shapes
:         └└╔
*tf.keras.backend.binary_crossentropy/mul_1Mul.tf.keras.backend.binary_crossentropy/sub_1:z:0.tf.keras.backend.binary_crossentropy/Log_1:y:0*
T0*-
_output_shapes
:         └└╔
*tf.keras.backend.binary_crossentropy/add_2AddV2,tf.keras.backend.binary_crossentropy/mul:z:0.tf.keras.backend.binary_crossentropy/mul_1:z:0*
T0*-
_output_shapes
:         └└Ч
(tf.keras.backend.binary_crossentropy/NegNeg.tf.keras.backend.binary_crossentropy/add_2:z:0*
T0*-
_output_shapes
:         └└z
tf.cast_1/CastCasttf.math.greater/Greater:z:0*

DstT0*

SrcT0
*-
_output_shapes
:         └└m
*tf.math.reduce_mean/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB o
,tf.math.reduce_mean/Mean/reduction_indices_1Const*
_output_shapes
: *
dtype0*
valueB ╜
tf.math.reduce_mean/MeanMean,tf.keras.backend.binary_crossentropy/Neg:y:05tf.math.reduce_mean/Mean/reduction_indices_1:output:0*
T0*-
_output_shapes
:         └└r
tf.math.equal/EqualEqualtf.cast_1/Cast:y:0inputs_2*
T0*-
_output_shapes
:         └└А
tf.math.multiply/MulMul!tf.math.reduce_mean/Mean:output:0inputs_1*
T0*-
_output_shapes
:         └└v
tf.cast_2/CastCasttf.math.equal/Equal:z:0*

DstT0*

SrcT0
*-
_output_shapes
:         └└y
(tf.math.reduce_sum/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      Ш
tf.math.reduce_sum/SumSumtf.math.multiply/Mul:z:01tf.math.reduce_sum/Sum/reduction_indices:output:0*
T0*#
_output_shapes
:         s
tf.math.multiply_1/MulMultf.cast_2/Cast:y:0inputs_1*
T0*-
_output_shapes
:         └└█
add_loss/PartitionedCallPartitionedCalltf.math.reduce_sum/Sum:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:         :         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_add_loss_layer_call_and_return_conditional_losses_4857{
*tf.math.reduce_sum_1/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      Ю
tf.math.reduce_sum_1/SumSumtf.math.multiply_1/Mul:z:03tf.math.reduce_sum_1/Sum/reduction_indices:output:0*
T0*#
_output_shapes
:         Е
"add_metric/StatefulPartitionedCallStatefulPartitionedCall!tf.math.reduce_sum_1/Sum:output:0add_metric_5094add_metric_5096*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_add_metric_layer_call_and_return_conditional_losses_4876l
IdentityIdentity!add_loss/PartitionedCall:output:1^NoOp*
T0*#
_output_shapes
:         Ш
NoOpNoOp#^add_metric/StatefulPartitionedCall)^conv2d_transpose/StatefulPartitionedCall+^conv2d_transpose_1/StatefulPartitionedCall!^my_dense/StatefulPartitionedCall)^separable_conv2d/StatefulPartitionedCall+^separable_conv2d_1/StatefulPartitionedCall+^separable_conv2d_2/StatefulPartitionedCall+^separable_conv2d_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*К
_input_shapesy
w:         └└:         └└:         └└: : : : : : : : : : : : : : : : : : : : 2H
"add_metric/StatefulPartitionedCall"add_metric/StatefulPartitionedCall2T
(conv2d_transpose/StatefulPartitionedCall(conv2d_transpose/StatefulPartitionedCall2X
*conv2d_transpose_1/StatefulPartitionedCall*conv2d_transpose_1/StatefulPartitionedCall2D
 my_dense/StatefulPartitionedCall my_dense/StatefulPartitionedCall2T
(separable_conv2d/StatefulPartitionedCall(separable_conv2d/StatefulPartitionedCall2X
*separable_conv2d_1/StatefulPartitionedCall*separable_conv2d_1/StatefulPartitionedCall2X
*separable_conv2d_2/StatefulPartitionedCall*separable_conv2d_2/StatefulPartitionedCall2X
*separable_conv2d_3/StatefulPartitionedCall*separable_conv2d_3/StatefulPartitionedCall:Y U
1
_output_shapes
:         └└
 
_user_specified_nameinputs:UQ
-
_output_shapes
:         └└
 
_user_specified_nameinputs:UQ
-
_output_shapes
:         └└
 
_user_specified_nameinputs
╫
B
&__inference_reshape_layer_call_fn_6139

inputs
identity▓
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:         └└* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_reshape_layer_call_and_return_conditional_losses_4819f
IdentityIdentityPartitionedCall:output:0*
T0*-
_output_shapes
:         └└"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+                           :i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
┌f
е

A__inference_model_1_layer_call_and_return_conditional_losses_5280

img_in

weights_in
true_labels/
separable_conv2d_5193:/
separable_conv2d_5195:#
separable_conv2d_5197:1
separable_conv2d_1_5200:1
separable_conv2d_1_5202:%
separable_conv2d_1_5204:1
separable_conv2d_2_5208:1
separable_conv2d_2_5210:%
separable_conv2d_2_5212:1
separable_conv2d_3_5215:1
separable_conv2d_3_5217:%
separable_conv2d_3_5219:/
conv2d_transpose_5223:#
conv2d_transpose_5225:1
conv2d_transpose_1_5228:%
conv2d_transpose_1_5230:
my_dense_5233:
my_dense_5235:
add_metric_5274: 
add_metric_5276: 
identityИв"add_metric/StatefulPartitionedCallв(conv2d_transpose/StatefulPartitionedCallв*conv2d_transpose_1/StatefulPartitionedCallв my_dense/StatefulPartitionedCallв(separable_conv2d/StatefulPartitionedCallв*separable_conv2d_1/StatefulPartitionedCallв*separable_conv2d_2/StatefulPartitionedCallв*separable_conv2d_3/StatefulPartitionedCallн
(separable_conv2d/StatefulPartitionedCallStatefulPartitionedCallimg_inseparable_conv2d_5193separable_conv2d_5195separable_conv2d_5197*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         └└*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_separable_conv2d_layer_call_and_return_conditional_losses_4501т
*separable_conv2d_1/StatefulPartitionedCallStatefulPartitionedCall1separable_conv2d/StatefulPartitionedCall:output:0separable_conv2d_1_5200separable_conv2d_1_5202separable_conv2d_1_5204*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         └└*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_separable_conv2d_1_layer_call_and_return_conditional_losses_4530ў
max_pooling2d/PartitionedCallPartitionedCall3separable_conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         аа* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_4548╫
*separable_conv2d_2/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0separable_conv2d_2_5208separable_conv2d_2_5210separable_conv2d_2_5212*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         аа*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_separable_conv2d_2_layer_call_and_return_conditional_losses_4571ф
*separable_conv2d_3/StatefulPartitionedCallStatefulPartitionedCall3separable_conv2d_2/StatefulPartitionedCall:output:0separable_conv2d_3_5215separable_conv2d_3_5217separable_conv2d_3_5219*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         аа*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_separable_conv2d_3_layer_call_and_return_conditional_losses_4600З
up_sampling2d/PartitionedCallPartitionedCall3separable_conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_up_sampling2d_layer_call_and_return_conditional_losses_4625─
(conv2d_transpose/StatefulPartitionedCallStatefulPartitionedCall&up_sampling2d/PartitionedCall:output:0conv2d_transpose_5223conv2d_transpose_5225*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_conv2d_transpose_layer_call_and_return_conditional_losses_4666╫
*conv2d_transpose_1/StatefulPartitionedCallStatefulPartitionedCall1conv2d_transpose/StatefulPartitionedCall:output:0conv2d_transpose_1_5228conv2d_transpose_1_5230*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_4711▒
 my_dense/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_1/StatefulPartitionedCall:output:0my_dense_5233my_dense_5235*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_my_dense_layer_call_and_return_conditional_losses_4800▌
reshape/PartitionedCallPartitionedCall)my_dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:         └└* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_reshape_layer_call_and_return_conditional_losses_4819^
tf.math.greater/Greater/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?а
tf.math.greater/GreaterGreater reshape/PartitionedCall:output:0"tf.math.greater/Greater/y:output:0*
T0*-
_output_shapes
:         └└o
*tf.keras.backend.binary_crossentropy/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Х┐╓3o
*tf.keras.backend.binary_crossentropy/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?║
(tf.keras.backend.binary_crossentropy/subSub3tf.keras.backend.binary_crossentropy/sub/x:output:03tf.keras.backend.binary_crossentropy/Const:output:0*
T0*
_output_shapes
: ═
:tf.keras.backend.binary_crossentropy/clip_by_value/MinimumMinimum reshape/PartitionedCall:output:0,tf.keras.backend.binary_crossentropy/sub:z:0*
T0*-
_output_shapes
:         └└ъ
2tf.keras.backend.binary_crossentropy/clip_by_valueMaximum>tf.keras.backend.binary_crossentropy/clip_by_value/Minimum:z:03tf.keras.backend.binary_crossentropy/Const:output:0*
T0*-
_output_shapes
:         └└o
*tf.keras.backend.binary_crossentropy/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Х┐╓3╓
(tf.keras.backend.binary_crossentropy/addAddV26tf.keras.backend.binary_crossentropy/clip_by_value:z:03tf.keras.backend.binary_crossentropy/add/y:output:0*
T0*-
_output_shapes
:         └└Х
(tf.keras.backend.binary_crossentropy/LogLog,tf.keras.backend.binary_crossentropy/add:z:0*
T0*-
_output_shapes
:         └└в
(tf.keras.backend.binary_crossentropy/mulMultrue_labels,tf.keras.backend.binary_crossentropy/Log:y:0*
T0*-
_output_shapes
:         └└q
,tf.keras.backend.binary_crossentropy/sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?н
*tf.keras.backend.binary_crossentropy/sub_1Sub5tf.keras.backend.binary_crossentropy/sub_1/x:output:0true_labels*
T0*-
_output_shapes
:         └└q
,tf.keras.backend.binary_crossentropy/sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╪
*tf.keras.backend.binary_crossentropy/sub_2Sub5tf.keras.backend.binary_crossentropy/sub_2/x:output:06tf.keras.backend.binary_crossentropy/clip_by_value:z:0*
T0*-
_output_shapes
:         └└q
,tf.keras.backend.binary_crossentropy/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *Х┐╓3╥
*tf.keras.backend.binary_crossentropy/add_1AddV2.tf.keras.backend.binary_crossentropy/sub_2:z:05tf.keras.backend.binary_crossentropy/add_1/y:output:0*
T0*-
_output_shapes
:         └└Щ
*tf.keras.backend.binary_crossentropy/Log_1Log.tf.keras.backend.binary_crossentropy/add_1:z:0*
T0*-
_output_shapes
:         └└╔
*tf.keras.backend.binary_crossentropy/mul_1Mul.tf.keras.backend.binary_crossentropy/sub_1:z:0.tf.keras.backend.binary_crossentropy/Log_1:y:0*
T0*-
_output_shapes
:         └└╔
*tf.keras.backend.binary_crossentropy/add_2AddV2,tf.keras.backend.binary_crossentropy/mul:z:0.tf.keras.backend.binary_crossentropy/mul_1:z:0*
T0*-
_output_shapes
:         └└Ч
(tf.keras.backend.binary_crossentropy/NegNeg.tf.keras.backend.binary_crossentropy/add_2:z:0*
T0*-
_output_shapes
:         └└z
tf.cast_1/CastCasttf.math.greater/Greater:z:0*

DstT0*

SrcT0
*-
_output_shapes
:         └└m
*tf.math.reduce_mean/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB o
,tf.math.reduce_mean/Mean/reduction_indices_1Const*
_output_shapes
: *
dtype0*
valueB ╜
tf.math.reduce_mean/MeanMean,tf.keras.backend.binary_crossentropy/Neg:y:05tf.math.reduce_mean/Mean/reduction_indices_1:output:0*
T0*-
_output_shapes
:         └└u
tf.math.equal/EqualEqualtf.cast_1/Cast:y:0true_labels*
T0*-
_output_shapes
:         └└В
tf.math.multiply/MulMul!tf.math.reduce_mean/Mean:output:0
weights_in*
T0*-
_output_shapes
:         └└v
tf.cast_2/CastCasttf.math.equal/Equal:z:0*

DstT0*

SrcT0
*-
_output_shapes
:         └└y
(tf.math.reduce_sum/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      Ш
tf.math.reduce_sum/SumSumtf.math.multiply/Mul:z:01tf.math.reduce_sum/Sum/reduction_indices:output:0*
T0*#
_output_shapes
:         u
tf.math.multiply_1/MulMultf.cast_2/Cast:y:0
weights_in*
T0*-
_output_shapes
:         └└█
add_loss/PartitionedCallPartitionedCalltf.math.reduce_sum/Sum:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:         :         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_add_loss_layer_call_and_return_conditional_losses_4857{
*tf.math.reduce_sum_1/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      Ю
tf.math.reduce_sum_1/SumSumtf.math.multiply_1/Mul:z:03tf.math.reduce_sum_1/Sum/reduction_indices:output:0*
T0*#
_output_shapes
:         Е
"add_metric/StatefulPartitionedCallStatefulPartitionedCall!tf.math.reduce_sum_1/Sum:output:0add_metric_5274add_metric_5276*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_add_metric_layer_call_and_return_conditional_losses_4876l
IdentityIdentity!add_loss/PartitionedCall:output:1^NoOp*
T0*#
_output_shapes
:         Ш
NoOpNoOp#^add_metric/StatefulPartitionedCall)^conv2d_transpose/StatefulPartitionedCall+^conv2d_transpose_1/StatefulPartitionedCall!^my_dense/StatefulPartitionedCall)^separable_conv2d/StatefulPartitionedCall+^separable_conv2d_1/StatefulPartitionedCall+^separable_conv2d_2/StatefulPartitionedCall+^separable_conv2d_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*К
_input_shapesy
w:         └└:         └└:         └└: : : : : : : : : : : : : : : : : : : : 2H
"add_metric/StatefulPartitionedCall"add_metric/StatefulPartitionedCall2T
(conv2d_transpose/StatefulPartitionedCall(conv2d_transpose/StatefulPartitionedCall2X
*conv2d_transpose_1/StatefulPartitionedCall*conv2d_transpose_1/StatefulPartitionedCall2D
 my_dense/StatefulPartitionedCall my_dense/StatefulPartitionedCall2T
(separable_conv2d/StatefulPartitionedCall(separable_conv2d/StatefulPartitionedCall2X
*separable_conv2d_1/StatefulPartitionedCall*separable_conv2d_1/StatefulPartitionedCall2X
*separable_conv2d_2/StatefulPartitionedCall*separable_conv2d_2/StatefulPartitionedCall2X
*separable_conv2d_3/StatefulPartitionedCall*separable_conv2d_3/StatefulPartitionedCall:Y U
1
_output_shapes
:         └└
 
_user_specified_nameimg_in:YU
-
_output_shapes
:         └└
$
_user_specified_name
weights_in:ZV
-
_output_shapes
:         └└
%
_user_specified_nametrue_labels
ч
n
B__inference_add_loss_layer_call_and_return_conditional_losses_4857

inputs
identity

identity_1J
IdentityIdentityinputs*
T0*#
_output_shapes
:         L

Identity_1Identityinputs*
T0*#
_output_shapes
:         "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*"
_input_shapes
:         :K G
#
_output_shapes
:         
 
_user_specified_nameinputs
┬!
Щ
L__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_4711

inputsB
(conv2d_transpose_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвconv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┘
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┘
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┘
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskР
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0▄
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+                           *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0Щ
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           j
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+                           {
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+                           Б
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+                           : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
┐f
а

A__inference_model_1_layer_call_and_return_conditional_losses_4883

inputs
inputs_1
inputs_2/
separable_conv2d_4729:/
separable_conv2d_4731:#
separable_conv2d_4733:1
separable_conv2d_1_4736:1
separable_conv2d_1_4738:%
separable_conv2d_1_4740:1
separable_conv2d_2_4744:1
separable_conv2d_2_4746:%
separable_conv2d_2_4748:1
separable_conv2d_3_4751:1
separable_conv2d_3_4753:%
separable_conv2d_3_4755:/
conv2d_transpose_4759:#
conv2d_transpose_4761:1
conv2d_transpose_1_4764:%
conv2d_transpose_1_4766:
my_dense_4801:
my_dense_4803:
add_metric_4877: 
add_metric_4879: 
identityИв"add_metric/StatefulPartitionedCallв(conv2d_transpose/StatefulPartitionedCallв*conv2d_transpose_1/StatefulPartitionedCallв my_dense/StatefulPartitionedCallв(separable_conv2d/StatefulPartitionedCallв*separable_conv2d_1/StatefulPartitionedCallв*separable_conv2d_2/StatefulPartitionedCallв*separable_conv2d_3/StatefulPartitionedCallн
(separable_conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsseparable_conv2d_4729separable_conv2d_4731separable_conv2d_4733*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         └└*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_separable_conv2d_layer_call_and_return_conditional_losses_4501т
*separable_conv2d_1/StatefulPartitionedCallStatefulPartitionedCall1separable_conv2d/StatefulPartitionedCall:output:0separable_conv2d_1_4736separable_conv2d_1_4738separable_conv2d_1_4740*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         └└*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_separable_conv2d_1_layer_call_and_return_conditional_losses_4530ў
max_pooling2d/PartitionedCallPartitionedCall3separable_conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         аа* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_4548╫
*separable_conv2d_2/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0separable_conv2d_2_4744separable_conv2d_2_4746separable_conv2d_2_4748*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         аа*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_separable_conv2d_2_layer_call_and_return_conditional_losses_4571ф
*separable_conv2d_3/StatefulPartitionedCallStatefulPartitionedCall3separable_conv2d_2/StatefulPartitionedCall:output:0separable_conv2d_3_4751separable_conv2d_3_4753separable_conv2d_3_4755*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         аа*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_separable_conv2d_3_layer_call_and_return_conditional_losses_4600З
up_sampling2d/PartitionedCallPartitionedCall3separable_conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_up_sampling2d_layer_call_and_return_conditional_losses_4625─
(conv2d_transpose/StatefulPartitionedCallStatefulPartitionedCall&up_sampling2d/PartitionedCall:output:0conv2d_transpose_4759conv2d_transpose_4761*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_conv2d_transpose_layer_call_and_return_conditional_losses_4666╫
*conv2d_transpose_1/StatefulPartitionedCallStatefulPartitionedCall1conv2d_transpose/StatefulPartitionedCall:output:0conv2d_transpose_1_4764conv2d_transpose_1_4766*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_4711▒
 my_dense/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_1/StatefulPartitionedCall:output:0my_dense_4801my_dense_4803*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_my_dense_layer_call_and_return_conditional_losses_4800▌
reshape/PartitionedCallPartitionedCall)my_dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:         └└* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_reshape_layer_call_and_return_conditional_losses_4819^
tf.math.greater/Greater/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?а
tf.math.greater/GreaterGreater reshape/PartitionedCall:output:0"tf.math.greater/Greater/y:output:0*
T0*-
_output_shapes
:         └└o
*tf.keras.backend.binary_crossentropy/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *Х┐╓3o
*tf.keras.backend.binary_crossentropy/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?║
(tf.keras.backend.binary_crossentropy/subSub3tf.keras.backend.binary_crossentropy/sub/x:output:03tf.keras.backend.binary_crossentropy/Const:output:0*
T0*
_output_shapes
: ═
:tf.keras.backend.binary_crossentropy/clip_by_value/MinimumMinimum reshape/PartitionedCall:output:0,tf.keras.backend.binary_crossentropy/sub:z:0*
T0*-
_output_shapes
:         └└ъ
2tf.keras.backend.binary_crossentropy/clip_by_valueMaximum>tf.keras.backend.binary_crossentropy/clip_by_value/Minimum:z:03tf.keras.backend.binary_crossentropy/Const:output:0*
T0*-
_output_shapes
:         └└o
*tf.keras.backend.binary_crossentropy/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Х┐╓3╓
(tf.keras.backend.binary_crossentropy/addAddV26tf.keras.backend.binary_crossentropy/clip_by_value:z:03tf.keras.backend.binary_crossentropy/add/y:output:0*
T0*-
_output_shapes
:         └└Х
(tf.keras.backend.binary_crossentropy/LogLog,tf.keras.backend.binary_crossentropy/add:z:0*
T0*-
_output_shapes
:         └└Я
(tf.keras.backend.binary_crossentropy/mulMulinputs_2,tf.keras.backend.binary_crossentropy/Log:y:0*
T0*-
_output_shapes
:         └└q
,tf.keras.backend.binary_crossentropy/sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?к
*tf.keras.backend.binary_crossentropy/sub_1Sub5tf.keras.backend.binary_crossentropy/sub_1/x:output:0inputs_2*
T0*-
_output_shapes
:         └└q
,tf.keras.backend.binary_crossentropy/sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╪
*tf.keras.backend.binary_crossentropy/sub_2Sub5tf.keras.backend.binary_crossentropy/sub_2/x:output:06tf.keras.backend.binary_crossentropy/clip_by_value:z:0*
T0*-
_output_shapes
:         └└q
,tf.keras.backend.binary_crossentropy/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *Х┐╓3╥
*tf.keras.backend.binary_crossentropy/add_1AddV2.tf.keras.backend.binary_crossentropy/sub_2:z:05tf.keras.backend.binary_crossentropy/add_1/y:output:0*
T0*-
_output_shapes
:         └└Щ
*tf.keras.backend.binary_crossentropy/Log_1Log.tf.keras.backend.binary_crossentropy/add_1:z:0*
T0*-
_output_shapes
:         └└╔
*tf.keras.backend.binary_crossentropy/mul_1Mul.tf.keras.backend.binary_crossentropy/sub_1:z:0.tf.keras.backend.binary_crossentropy/Log_1:y:0*
T0*-
_output_shapes
:         └└╔
*tf.keras.backend.binary_crossentropy/add_2AddV2,tf.keras.backend.binary_crossentropy/mul:z:0.tf.keras.backend.binary_crossentropy/mul_1:z:0*
T0*-
_output_shapes
:         └└Ч
(tf.keras.backend.binary_crossentropy/NegNeg.tf.keras.backend.binary_crossentropy/add_2:z:0*
T0*-
_output_shapes
:         └└z
tf.cast_1/CastCasttf.math.greater/Greater:z:0*

DstT0*

SrcT0
*-
_output_shapes
:         └└m
*tf.math.reduce_mean/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB o
,tf.math.reduce_mean/Mean/reduction_indices_1Const*
_output_shapes
: *
dtype0*
valueB ╜
tf.math.reduce_mean/MeanMean,tf.keras.backend.binary_crossentropy/Neg:y:05tf.math.reduce_mean/Mean/reduction_indices_1:output:0*
T0*-
_output_shapes
:         └└r
tf.math.equal/EqualEqualtf.cast_1/Cast:y:0inputs_2*
T0*-
_output_shapes
:         └└А
tf.math.multiply/MulMul!tf.math.reduce_mean/Mean:output:0inputs_1*
T0*-
_output_shapes
:         └└v
tf.cast_2/CastCasttf.math.equal/Equal:z:0*

DstT0*

SrcT0
*-
_output_shapes
:         └└y
(tf.math.reduce_sum/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      Ш
tf.math.reduce_sum/SumSumtf.math.multiply/Mul:z:01tf.math.reduce_sum/Sum/reduction_indices:output:0*
T0*#
_output_shapes
:         s
tf.math.multiply_1/MulMultf.cast_2/Cast:y:0inputs_1*
T0*-
_output_shapes
:         └└█
add_loss/PartitionedCallPartitionedCalltf.math.reduce_sum/Sum:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:         :         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_add_loss_layer_call_and_return_conditional_losses_4857{
*tf.math.reduce_sum_1/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      Ю
tf.math.reduce_sum_1/SumSumtf.math.multiply_1/Mul:z:03tf.math.reduce_sum_1/Sum/reduction_indices:output:0*
T0*#
_output_shapes
:         Е
"add_metric/StatefulPartitionedCallStatefulPartitionedCall!tf.math.reduce_sum_1/Sum:output:0add_metric_4877add_metric_4879*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_add_metric_layer_call_and_return_conditional_losses_4876l
IdentityIdentity!add_loss/PartitionedCall:output:1^NoOp*
T0*#
_output_shapes
:         Ш
NoOpNoOp#^add_metric/StatefulPartitionedCall)^conv2d_transpose/StatefulPartitionedCall+^conv2d_transpose_1/StatefulPartitionedCall!^my_dense/StatefulPartitionedCall)^separable_conv2d/StatefulPartitionedCall+^separable_conv2d_1/StatefulPartitionedCall+^separable_conv2d_2/StatefulPartitionedCall+^separable_conv2d_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*К
_input_shapesy
w:         └└:         └└:         └└: : : : : : : : : : : : : : : : : : : : 2H
"add_metric/StatefulPartitionedCall"add_metric/StatefulPartitionedCall2T
(conv2d_transpose/StatefulPartitionedCall(conv2d_transpose/StatefulPartitionedCall2X
*conv2d_transpose_1/StatefulPartitionedCall*conv2d_transpose_1/StatefulPartitionedCall2D
 my_dense/StatefulPartitionedCall my_dense/StatefulPartitionedCall2T
(separable_conv2d/StatefulPartitionedCall(separable_conv2d/StatefulPartitionedCall2X
*separable_conv2d_1/StatefulPartitionedCall*separable_conv2d_1/StatefulPartitionedCall2X
*separable_conv2d_2/StatefulPartitionedCall*separable_conv2d_2/StatefulPartitionedCall2X
*separable_conv2d_3/StatefulPartitionedCall*separable_conv2d_3/StatefulPartitionedCall:Y U
1
_output_shapes
:         └└
 
_user_specified_nameinputs:UQ
-
_output_shapes
:         └└
 
_user_specified_nameinputs:UQ
-
_output_shapes
:         └└
 
_user_specified_nameinputs
О
c
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_4548

inputs
identityб
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingSAME*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
┐
д
/__inference_conv2d_transpose_layer_call_fn_6017

inputs!
unknown:
	unknown_0:
identityИвStatefulPartitionedCall∙
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_conv2d_transpose_layer_call_and_return_conditional_losses_4666Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+                           : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
Н
]
A__inference_reshape_layer_call_and_return_conditional_losses_4819

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskR
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :└R
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :└П
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:j
ReshapeReshapeinputsReshape/shape:output:0*
T0*-
_output_shapes
:         └└^
IdentityIdentityReshape:output:0*
T0*-
_output_shapes
:         └└"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+                           :i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
ў
╦
1__inference_separable_conv2d_3_layer_call_fn_5975

inputs!
unknown:#
	unknown_0:
	unknown_1:
identityИвStatefulPartitionedCallИ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_separable_conv2d_3_layer_call_and_return_conditional_losses_4600Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:+                           : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
└
∙
B__inference_my_dense_layer_call_and_return_conditional_losses_6134

inputs3
!tensordot_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвTensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:c
Tensordot/freeConst*
_output_shapes
:*
dtype0*!
valueB"          E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ╗
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ┐
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ь
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:П
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*A
_output_shapes/
-:+                           К
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:                  К
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         [
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : з
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Щ
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*A
_output_shapes/
-:+                           r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0Т
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           p
SigmoidSigmoidBiasAdd:output:0*
T0*A
_output_shapes/
-:+                           t
IdentityIdentitySigmoid:y:0^NoOp*
T0*A
_output_shapes/
-:+                           z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+                           : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
Щ
C
'__inference_add_loss_layer_call_fn_6158

inputs
identity╣
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *2
_output_shapes 
:         :         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_add_loss_layer_call_and_return_conditional_losses_4857\
IdentityIdentityPartitionedCall:output:0*
T0*#
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*"
_input_shapes
:         :K G
#
_output_shapes
:         
 
_user_specified_nameinputs
К
▌
"__inference_signature_wrapper_5425

img_in
true_labels

weights_in!
unknown:#
	unknown_0:
	unknown_1:#
	unknown_2:#
	unknown_3:
	unknown_4:#
	unknown_5:#
	unknown_6:
	unknown_7:#
	unknown_8:#
	unknown_9:

unknown_10:$

unknown_11:

unknown_12:$

unknown_13:

unknown_14:

unknown_15:

unknown_16:

unknown_17: 

unknown_18: ИвStatefulPartitionedCallк
StatefulPartitionedCallStatefulPartitionedCallimg_in
weights_intrue_labelsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18*"
Tin
2*

Tout
 *
_collective_manager_ids
 *
_output_shapes
 *4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *(
f#R!
__inference__wrapped_model_4481*(
_construction_contextkEagerRuntime*К
_input_shapesy
w:         └└:         └└:         └└: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:         └└
 
_user_specified_nameimg_in:ZV
-
_output_shapes
:         └└
%
_user_specified_nametrue_labels:YU
-
_output_shapes
:         └└
$
_user_specified_name
weights_in
┬!
Щ
L__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_6094

inputsB
(conv2d_transpose_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвconv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┘
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┘
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┘
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskР
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0▄
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+                           *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0Щ
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           j
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+                           {
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+                           Б
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+                           : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
О
c
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_5937

inputs
identityб
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingSAME*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
к
Г
L__inference_separable_conv2d_2_layer_call_and_return_conditional_losses_5964

inputsB
(separable_conv2d_readvariableop_resource:D
*separable_conv2d_readvariableop_1_resource:-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвseparable_conv2d/ReadVariableOpв!separable_conv2d/ReadVariableOp_1Р
separable_conv2d/ReadVariableOpReadVariableOp(separable_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ф
!separable_conv2d/ReadVariableOp_1ReadVariableOp*separable_conv2d_readvariableop_1_resource*&
_output_shapes
:*
dtype0o
separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            o
separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      ╪
separable_conv2d/depthwiseDepthwiseConv2dNativeinputs'separable_conv2d/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           *
paddingSAME*
strides
▀
separable_conv2dConv2D#separable_conv2d/depthwise:output:0)separable_conv2d/ReadVariableOp_1:value:0*
T0*A
_output_shapes/
-:+                           *
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0Щ
BiasAddBiasAddseparable_conv2d:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           j
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+                           {
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+                           е
NoOpNoOp^BiasAdd/ReadVariableOp ^separable_conv2d/ReadVariableOp"^separable_conv2d/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:+                           : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
separable_conv2d/ReadVariableOpseparable_conv2d/ReadVariableOp2F
!separable_conv2d/ReadVariableOp_1!separable_conv2d/ReadVariableOp_1:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
и
Б
J__inference_separable_conv2d_layer_call_and_return_conditional_losses_4501

inputsB
(separable_conv2d_readvariableop_resource:D
*separable_conv2d_readvariableop_1_resource:-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвseparable_conv2d/ReadVariableOpв!separable_conv2d/ReadVariableOp_1Р
separable_conv2d/ReadVariableOpReadVariableOp(separable_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ф
!separable_conv2d/ReadVariableOp_1ReadVariableOp*separable_conv2d_readvariableop_1_resource*&
_output_shapes
:*
dtype0o
separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            o
separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      ╪
separable_conv2d/depthwiseDepthwiseConv2dNativeinputs'separable_conv2d/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           *
paddingSAME*
strides
▀
separable_conv2dConv2D#separable_conv2d/depthwise:output:0)separable_conv2d/ReadVariableOp_1:value:0*
T0*A
_output_shapes/
-:+                           *
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0Щ
BiasAddBiasAddseparable_conv2d:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           j
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+                           {
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+                           е
NoOpNoOp^BiasAdd/ReadVariableOp ^separable_conv2d/ReadVariableOp"^separable_conv2d/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:+                           : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
separable_conv2d/ReadVariableOpseparable_conv2d/ReadVariableOp2F
!separable_conv2d/ReadVariableOp_1!separable_conv2d/ReadVariableOp_1:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
╣
▐
&__inference_model_1_layer_call_fn_5471
inputs_0
inputs_1
inputs_2!
unknown:#
	unknown_0:
	unknown_1:#
	unknown_2:#
	unknown_3:
	unknown_4:#
	unknown_5:#
	unknown_6:
	unknown_7:#
	unknown_8:#
	unknown_9:

unknown_10:$

unknown_11:

unknown_12:$

unknown_13:

unknown_14:

unknown_15:

unknown_16:

unknown_17: 

unknown_18: ИвStatefulPartitionedCall█
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1inputs_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18*"
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:         *4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_model_1_layer_call_and_return_conditional_losses_4883*(
_construction_contextkEagerRuntime*К
_input_shapesy
w:         └└:         └└:         └└: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
1
_output_shapes
:         └└
"
_user_specified_name
inputs/0:WS
-
_output_shapes
:         └└
"
_user_specified_name
inputs/1:WS
-
_output_shapes
:         └└
"
_user_specified_name
inputs/2
чИ
╪-
 __inference__traced_restore_6598
file_prefixL
2assignvariableop_separable_conv2d_depthwise_kernel:N
4assignvariableop_1_separable_conv2d_pointwise_kernel:6
(assignvariableop_2_separable_conv2d_bias:P
6assignvariableop_3_separable_conv2d_1_depthwise_kernel:P
6assignvariableop_4_separable_conv2d_1_pointwise_kernel:8
*assignvariableop_5_separable_conv2d_1_bias:P
6assignvariableop_6_separable_conv2d_2_depthwise_kernel:P
6assignvariableop_7_separable_conv2d_2_pointwise_kernel:8
*assignvariableop_8_separable_conv2d_2_bias:P
6assignvariableop_9_separable_conv2d_3_depthwise_kernel:Q
7assignvariableop_10_separable_conv2d_3_pointwise_kernel:9
+assignvariableop_11_separable_conv2d_3_bias:E
+assignvariableop_12_conv2d_transpose_kernel:7
)assignvariableop_13_conv2d_transpose_bias:G
-assignvariableop_14_conv2d_transpose_1_kernel:9
+assignvariableop_15_conv2d_transpose_1_bias:5
#assignvariableop_16_my_dense_kernel:/
!assignvariableop_17_my_dense_bias:'
assignvariableop_18_adam_iter:	 )
assignvariableop_19_adam_beta_1: )
assignvariableop_20_adam_beta_2: (
assignvariableop_21_adam_decay: 0
&assignvariableop_22_adam_learning_rate: #
assignvariableop_23_total: #
assignvariableop_24_count: .
$assignvariableop_25_add_metric_total: .
$assignvariableop_26_add_metric_count: V
<assignvariableop_27_adam_separable_conv2d_depthwise_kernel_m:V
<assignvariableop_28_adam_separable_conv2d_pointwise_kernel_m:>
0assignvariableop_29_adam_separable_conv2d_bias_m:X
>assignvariableop_30_adam_separable_conv2d_1_depthwise_kernel_m:X
>assignvariableop_31_adam_separable_conv2d_1_pointwise_kernel_m:@
2assignvariableop_32_adam_separable_conv2d_1_bias_m:X
>assignvariableop_33_adam_separable_conv2d_2_depthwise_kernel_m:X
>assignvariableop_34_adam_separable_conv2d_2_pointwise_kernel_m:@
2assignvariableop_35_adam_separable_conv2d_2_bias_m:X
>assignvariableop_36_adam_separable_conv2d_3_depthwise_kernel_m:X
>assignvariableop_37_adam_separable_conv2d_3_pointwise_kernel_m:@
2assignvariableop_38_adam_separable_conv2d_3_bias_m:L
2assignvariableop_39_adam_conv2d_transpose_kernel_m:>
0assignvariableop_40_adam_conv2d_transpose_bias_m:N
4assignvariableop_41_adam_conv2d_transpose_1_kernel_m:@
2assignvariableop_42_adam_conv2d_transpose_1_bias_m:<
*assignvariableop_43_adam_my_dense_kernel_m:6
(assignvariableop_44_adam_my_dense_bias_m:V
<assignvariableop_45_adam_separable_conv2d_depthwise_kernel_v:V
<assignvariableop_46_adam_separable_conv2d_pointwise_kernel_v:>
0assignvariableop_47_adam_separable_conv2d_bias_v:X
>assignvariableop_48_adam_separable_conv2d_1_depthwise_kernel_v:X
>assignvariableop_49_adam_separable_conv2d_1_pointwise_kernel_v:@
2assignvariableop_50_adam_separable_conv2d_1_bias_v:X
>assignvariableop_51_adam_separable_conv2d_2_depthwise_kernel_v:X
>assignvariableop_52_adam_separable_conv2d_2_pointwise_kernel_v:@
2assignvariableop_53_adam_separable_conv2d_2_bias_v:X
>assignvariableop_54_adam_separable_conv2d_3_depthwise_kernel_v:X
>assignvariableop_55_adam_separable_conv2d_3_pointwise_kernel_v:@
2assignvariableop_56_adam_separable_conv2d_3_bias_v:L
2assignvariableop_57_adam_conv2d_transpose_kernel_v:>
0assignvariableop_58_adam_conv2d_transpose_bias_v:N
4assignvariableop_59_adam_conv2d_transpose_1_kernel_v:@
2assignvariableop_60_adam_conv2d_transpose_1_bias_v:<
*assignvariableop_61_adam_my_dense_kernel_v:6
(assignvariableop_62_adam_my_dense_bias_v:
identity_64ИвAssignVariableOpвAssignVariableOp_1вAssignVariableOp_10вAssignVariableOp_11вAssignVariableOp_12вAssignVariableOp_13вAssignVariableOp_14вAssignVariableOp_15вAssignVariableOp_16вAssignVariableOp_17вAssignVariableOp_18вAssignVariableOp_19вAssignVariableOp_2вAssignVariableOp_20вAssignVariableOp_21вAssignVariableOp_22вAssignVariableOp_23вAssignVariableOp_24вAssignVariableOp_25вAssignVariableOp_26вAssignVariableOp_27вAssignVariableOp_28вAssignVariableOp_29вAssignVariableOp_3вAssignVariableOp_30вAssignVariableOp_31вAssignVariableOp_32вAssignVariableOp_33вAssignVariableOp_34вAssignVariableOp_35вAssignVariableOp_36вAssignVariableOp_37вAssignVariableOp_38вAssignVariableOp_39вAssignVariableOp_4вAssignVariableOp_40вAssignVariableOp_41вAssignVariableOp_42вAssignVariableOp_43вAssignVariableOp_44вAssignVariableOp_45вAssignVariableOp_46вAssignVariableOp_47вAssignVariableOp_48вAssignVariableOp_49вAssignVariableOp_5вAssignVariableOp_50вAssignVariableOp_51вAssignVariableOp_52вAssignVariableOp_53вAssignVariableOp_54вAssignVariableOp_55вAssignVariableOp_56вAssignVariableOp_57вAssignVariableOp_58вAssignVariableOp_59вAssignVariableOp_6вAssignVariableOp_60вAssignVariableOp_61вAssignVariableOp_62вAssignVariableOp_7вAssignVariableOp_8вAssignVariableOp_9╨%
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:@*
dtype0*Ў$
valueь$Bщ$@B@layer_with_weights-0/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-0/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-1/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-1/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-2/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-2/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-3/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-3/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-0/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-0/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-1/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-1/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-2/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-2/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-3/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-3/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-0/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-0/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-1/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-1/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-2/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-2/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-3/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-3/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHє
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:@*
dtype0*Х
valueЛBИ@B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B с
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Ц
_output_shapesГ
А::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*N
dtypesD
B2@	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:Э
AssignVariableOpAssignVariableOp2assignvariableop_separable_conv2d_depthwise_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:г
AssignVariableOp_1AssignVariableOp4assignvariableop_1_separable_conv2d_pointwise_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:Ч
AssignVariableOp_2AssignVariableOp(assignvariableop_2_separable_conv2d_biasIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:е
AssignVariableOp_3AssignVariableOp6assignvariableop_3_separable_conv2d_1_depthwise_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:е
AssignVariableOp_4AssignVariableOp6assignvariableop_4_separable_conv2d_1_pointwise_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_5AssignVariableOp*assignvariableop_5_separable_conv2d_1_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:е
AssignVariableOp_6AssignVariableOp6assignvariableop_6_separable_conv2d_2_depthwise_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:е
AssignVariableOp_7AssignVariableOp6assignvariableop_7_separable_conv2d_2_pointwise_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_8AssignVariableOp*assignvariableop_8_separable_conv2d_2_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:е
AssignVariableOp_9AssignVariableOp6assignvariableop_9_separable_conv2d_3_depthwise_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:и
AssignVariableOp_10AssignVariableOp7assignvariableop_10_separable_conv2d_3_pointwise_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:Ь
AssignVariableOp_11AssignVariableOp+assignvariableop_11_separable_conv2d_3_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:Ь
AssignVariableOp_12AssignVariableOp+assignvariableop_12_conv2d_transpose_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:Ъ
AssignVariableOp_13AssignVariableOp)assignvariableop_13_conv2d_transpose_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:Ю
AssignVariableOp_14AssignVariableOp-assignvariableop_14_conv2d_transpose_1_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:Ь
AssignVariableOp_15AssignVariableOp+assignvariableop_15_conv2d_transpose_1_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:Ф
AssignVariableOp_16AssignVariableOp#assignvariableop_16_my_dense_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:Т
AssignVariableOp_17AssignVariableOp!assignvariableop_17_my_dense_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0	*
_output_shapes
:О
AssignVariableOp_18AssignVariableOpassignvariableop_18_adam_iterIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_19AssignVariableOpassignvariableop_19_adam_beta_1Identity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_20AssignVariableOpassignvariableop_20_adam_beta_2Identity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:П
AssignVariableOp_21AssignVariableOpassignvariableop_21_adam_decayIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:Ч
AssignVariableOp_22AssignVariableOp&assignvariableop_22_adam_learning_rateIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_23AssignVariableOpassignvariableop_23_totalIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_24AssignVariableOpassignvariableop_24_countIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:Х
AssignVariableOp_25AssignVariableOp$assignvariableop_25_add_metric_totalIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:Х
AssignVariableOp_26AssignVariableOp$assignvariableop_26_add_metric_countIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:н
AssignVariableOp_27AssignVariableOp<assignvariableop_27_adam_separable_conv2d_depthwise_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:н
AssignVariableOp_28AssignVariableOp<assignvariableop_28_adam_separable_conv2d_pointwise_kernel_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:б
AssignVariableOp_29AssignVariableOp0assignvariableop_29_adam_separable_conv2d_bias_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:п
AssignVariableOp_30AssignVariableOp>assignvariableop_30_adam_separable_conv2d_1_depthwise_kernel_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:п
AssignVariableOp_31AssignVariableOp>assignvariableop_31_adam_separable_conv2d_1_pointwise_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:г
AssignVariableOp_32AssignVariableOp2assignvariableop_32_adam_separable_conv2d_1_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:п
AssignVariableOp_33AssignVariableOp>assignvariableop_33_adam_separable_conv2d_2_depthwise_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:п
AssignVariableOp_34AssignVariableOp>assignvariableop_34_adam_separable_conv2d_2_pointwise_kernel_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:г
AssignVariableOp_35AssignVariableOp2assignvariableop_35_adam_separable_conv2d_2_bias_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:п
AssignVariableOp_36AssignVariableOp>assignvariableop_36_adam_separable_conv2d_3_depthwise_kernel_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:п
AssignVariableOp_37AssignVariableOp>assignvariableop_37_adam_separable_conv2d_3_pointwise_kernel_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:г
AssignVariableOp_38AssignVariableOp2assignvariableop_38_adam_separable_conv2d_3_bias_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:г
AssignVariableOp_39AssignVariableOp2assignvariableop_39_adam_conv2d_transpose_kernel_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:б
AssignVariableOp_40AssignVariableOp0assignvariableop_40_adam_conv2d_transpose_bias_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:е
AssignVariableOp_41AssignVariableOp4assignvariableop_41_adam_conv2d_transpose_1_kernel_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:г
AssignVariableOp_42AssignVariableOp2assignvariableop_42_adam_conv2d_transpose_1_bias_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_43AssignVariableOp*assignvariableop_43_adam_my_dense_kernel_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_44AssignVariableOp(assignvariableop_44_adam_my_dense_bias_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:н
AssignVariableOp_45AssignVariableOp<assignvariableop_45_adam_separable_conv2d_depthwise_kernel_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:н
AssignVariableOp_46AssignVariableOp<assignvariableop_46_adam_separable_conv2d_pointwise_kernel_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:б
AssignVariableOp_47AssignVariableOp0assignvariableop_47_adam_separable_conv2d_bias_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:п
AssignVariableOp_48AssignVariableOp>assignvariableop_48_adam_separable_conv2d_1_depthwise_kernel_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:п
AssignVariableOp_49AssignVariableOp>assignvariableop_49_adam_separable_conv2d_1_pointwise_kernel_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:г
AssignVariableOp_50AssignVariableOp2assignvariableop_50_adam_separable_conv2d_1_bias_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:п
AssignVariableOp_51AssignVariableOp>assignvariableop_51_adam_separable_conv2d_2_depthwise_kernel_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:п
AssignVariableOp_52AssignVariableOp>assignvariableop_52_adam_separable_conv2d_2_pointwise_kernel_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:г
AssignVariableOp_53AssignVariableOp2assignvariableop_53_adam_separable_conv2d_2_bias_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:п
AssignVariableOp_54AssignVariableOp>assignvariableop_54_adam_separable_conv2d_3_depthwise_kernel_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:п
AssignVariableOp_55AssignVariableOp>assignvariableop_55_adam_separable_conv2d_3_pointwise_kernel_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:г
AssignVariableOp_56AssignVariableOp2assignvariableop_56_adam_separable_conv2d_3_bias_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:г
AssignVariableOp_57AssignVariableOp2assignvariableop_57_adam_conv2d_transpose_kernel_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:б
AssignVariableOp_58AssignVariableOp0assignvariableop_58_adam_conv2d_transpose_bias_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:е
AssignVariableOp_59AssignVariableOp4assignvariableop_59_adam_conv2d_transpose_1_kernel_vIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:г
AssignVariableOp_60AssignVariableOp2assignvariableop_60_adam_conv2d_transpose_1_bias_vIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_61AssignVariableOp*assignvariableop_61_adam_my_dense_kernel_vIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_62AssignVariableOp(assignvariableop_62_adam_my_dense_bias_vIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ╣
Identity_63Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_64IdentityIdentity_63:output:0^NoOp_1*
T0*
_output_shapes
: ж
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_64Identity_64:output:0*Х
_input_shapesГ
А: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
з
Ф
'__inference_my_dense_layer_call_fn_6103

inputs
unknown:
	unknown_0:
identityИвStatefulPartitionedCallё
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_my_dense_layer_call_and_return_conditional_losses_4800Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+                           : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
└!
Ч
J__inference_conv2d_transpose_layer_call_and_return_conditional_losses_6051

inputsB
(conv2d_transpose_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвconv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┘
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┘
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┘
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskР
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0▄
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+                           *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0Щ
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           j
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+                           {
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+                           Б
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+                           : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
╒
Я
D__inference_add_metric_layer_call_and_return_conditional_losses_6186

inputs&
assignaddvariableop_resource: (
assignaddvariableop_1_resource: 

identity_1ИвAssignAddVariableOpвAssignAddVariableOp_1вdiv_no_nan/ReadVariableOpвdiv_no_nan/ReadVariableOp_1O
ConstConst*
_output_shapes
:*
dtype0*
valueB: C
SumSuminputsConst:output:0*
T0*
_output_shapes
: y
AssignAddVariableOpAssignAddVariableOpassignaddvariableop_resourceSum:output:0*
_output_shapes
 *
dtype05
SizeSizeinputs*
T0*
_output_shapes
: K
CastCastSize:output:0*

DstT0*

SrcT0*
_output_shapes
: П
AssignAddVariableOp_1AssignAddVariableOpassignaddvariableop_1_resourceCast:y:0^AssignAddVariableOp*
_output_shapes
 *
dtype0Ь
div_no_nan/ReadVariableOpReadVariableOpassignaddvariableop_resource^AssignAddVariableOp^AssignAddVariableOp_1*
_output_shapes
: *
dtype0К
div_no_nan/ReadVariableOp_1ReadVariableOpassignaddvariableop_1_resource^AssignAddVariableOp_1*
_output_shapes
: *
dtype0

div_no_nanDivNoNan!div_no_nan/ReadVariableOp:value:0#div_no_nan/ReadVariableOp_1:value:0*
T0*
_output_shapes
: E
IdentityIdentitydiv_no_nan:z:0*
T0*
_output_shapes
: S

Identity_1Identityinputs^NoOp*
T0*#
_output_shapes
:         о
NoOpNoOp^AssignAddVariableOp^AssignAddVariableOp_1^div_no_nan/ReadVariableOp^div_no_nan/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         : : 2*
AssignAddVariableOpAssignAddVariableOp2.
AssignAddVariableOp_1AssignAddVariableOp_126
div_no_nan/ReadVariableOpdiv_no_nan/ReadVariableOp2:
div_no_nan/ReadVariableOp_1div_no_nan/ReadVariableOp_1:K G
#
_output_shapes
:         
 
_user_specified_nameinputs
ў
╦
1__inference_separable_conv2d_1_layer_call_fn_5911

inputs!
unknown:#
	unknown_0:
	unknown_1:
identityИвStatefulPartitionedCallИ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_separable_conv2d_1_layer_call_and_return_conditional_losses_4530Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:+                           : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
и
Б
J__inference_separable_conv2d_layer_call_and_return_conditional_losses_5900

inputsB
(separable_conv2d_readvariableop_resource:D
*separable_conv2d_readvariableop_1_resource:-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвseparable_conv2d/ReadVariableOpв!separable_conv2d/ReadVariableOp_1Р
separable_conv2d/ReadVariableOpReadVariableOp(separable_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ф
!separable_conv2d/ReadVariableOp_1ReadVariableOp*separable_conv2d_readvariableop_1_resource*&
_output_shapes
:*
dtype0o
separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            o
separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      ╪
separable_conv2d/depthwiseDepthwiseConv2dNativeinputs'separable_conv2d/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           *
paddingSAME*
strides
▀
separable_conv2dConv2D#separable_conv2d/depthwise:output:0)separable_conv2d/ReadVariableOp_1:value:0*
T0*A
_output_shapes/
-:+                           *
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0Щ
BiasAddBiasAddseparable_conv2d:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           j
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+                           {
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+                           е
NoOpNoOp^BiasAdd/ReadVariableOp ^separable_conv2d/ReadVariableOp"^separable_conv2d/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:+                           : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
separable_conv2d/ReadVariableOpseparable_conv2d/ReadVariableOp2F
!separable_conv2d/ReadVariableOp_1!separable_conv2d/ReadVariableOp_1:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
в
К
)__inference_add_metric_layer_call_fn_6172

inputs
unknown: 
	unknown_0: 
identityИвStatefulPartitionedCall╤
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_add_metric_layer_call_and_return_conditional_losses_4876k
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*#
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         : : 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:         
 
_user_specified_nameinputs
┬
с
&__inference_model_1_layer_call_fn_4925

img_in

weights_in
true_labels!
unknown:#
	unknown_0:
	unknown_1:#
	unknown_2:#
	unknown_3:
	unknown_4:#
	unknown_5:#
	unknown_6:
	unknown_7:#
	unknown_8:#
	unknown_9:

unknown_10:$

unknown_11:

unknown_12:$

unknown_13:

unknown_14:

unknown_15:

unknown_16:

unknown_17: 

unknown_18: ИвStatefulPartitionedCall▐
StatefulPartitionedCallStatefulPartitionedCallimg_in
weights_intrue_labelsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18*"
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:         *4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_model_1_layer_call_and_return_conditional_losses_4883*(
_construction_contextkEagerRuntime*К
_input_shapesy
w:         └└:         └└:         └└: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:         └└
 
_user_specified_nameimg_in:YU
-
_output_shapes
:         └└
$
_user_specified_name
weights_in:ZV
-
_output_shapes
:         └└
%
_user_specified_nametrue_labels
ўК
Ь
__inference__traced_save_6399
file_prefix@
<savev2_separable_conv2d_depthwise_kernel_read_readvariableop@
<savev2_separable_conv2d_pointwise_kernel_read_readvariableop4
0savev2_separable_conv2d_bias_read_readvariableopB
>savev2_separable_conv2d_1_depthwise_kernel_read_readvariableopB
>savev2_separable_conv2d_1_pointwise_kernel_read_readvariableop6
2savev2_separable_conv2d_1_bias_read_readvariableopB
>savev2_separable_conv2d_2_depthwise_kernel_read_readvariableopB
>savev2_separable_conv2d_2_pointwise_kernel_read_readvariableop6
2savev2_separable_conv2d_2_bias_read_readvariableopB
>savev2_separable_conv2d_3_depthwise_kernel_read_readvariableopB
>savev2_separable_conv2d_3_pointwise_kernel_read_readvariableop6
2savev2_separable_conv2d_3_bias_read_readvariableop6
2savev2_conv2d_transpose_kernel_read_readvariableop4
0savev2_conv2d_transpose_bias_read_readvariableop8
4savev2_conv2d_transpose_1_kernel_read_readvariableop6
2savev2_conv2d_transpose_1_bias_read_readvariableop.
*savev2_my_dense_kernel_read_readvariableop,
(savev2_my_dense_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop/
+savev2_add_metric_total_read_readvariableop/
+savev2_add_metric_count_read_readvariableopG
Csavev2_adam_separable_conv2d_depthwise_kernel_m_read_readvariableopG
Csavev2_adam_separable_conv2d_pointwise_kernel_m_read_readvariableop;
7savev2_adam_separable_conv2d_bias_m_read_readvariableopI
Esavev2_adam_separable_conv2d_1_depthwise_kernel_m_read_readvariableopI
Esavev2_adam_separable_conv2d_1_pointwise_kernel_m_read_readvariableop=
9savev2_adam_separable_conv2d_1_bias_m_read_readvariableopI
Esavev2_adam_separable_conv2d_2_depthwise_kernel_m_read_readvariableopI
Esavev2_adam_separable_conv2d_2_pointwise_kernel_m_read_readvariableop=
9savev2_adam_separable_conv2d_2_bias_m_read_readvariableopI
Esavev2_adam_separable_conv2d_3_depthwise_kernel_m_read_readvariableopI
Esavev2_adam_separable_conv2d_3_pointwise_kernel_m_read_readvariableop=
9savev2_adam_separable_conv2d_3_bias_m_read_readvariableop=
9savev2_adam_conv2d_transpose_kernel_m_read_readvariableop;
7savev2_adam_conv2d_transpose_bias_m_read_readvariableop?
;savev2_adam_conv2d_transpose_1_kernel_m_read_readvariableop=
9savev2_adam_conv2d_transpose_1_bias_m_read_readvariableop5
1savev2_adam_my_dense_kernel_m_read_readvariableop3
/savev2_adam_my_dense_bias_m_read_readvariableopG
Csavev2_adam_separable_conv2d_depthwise_kernel_v_read_readvariableopG
Csavev2_adam_separable_conv2d_pointwise_kernel_v_read_readvariableop;
7savev2_adam_separable_conv2d_bias_v_read_readvariableopI
Esavev2_adam_separable_conv2d_1_depthwise_kernel_v_read_readvariableopI
Esavev2_adam_separable_conv2d_1_pointwise_kernel_v_read_readvariableop=
9savev2_adam_separable_conv2d_1_bias_v_read_readvariableopI
Esavev2_adam_separable_conv2d_2_depthwise_kernel_v_read_readvariableopI
Esavev2_adam_separable_conv2d_2_pointwise_kernel_v_read_readvariableop=
9savev2_adam_separable_conv2d_2_bias_v_read_readvariableopI
Esavev2_adam_separable_conv2d_3_depthwise_kernel_v_read_readvariableopI
Esavev2_adam_separable_conv2d_3_pointwise_kernel_v_read_readvariableop=
9savev2_adam_separable_conv2d_3_bias_v_read_readvariableop=
9savev2_adam_conv2d_transpose_kernel_v_read_readvariableop;
7savev2_adam_conv2d_transpose_bias_v_read_readvariableop?
;savev2_adam_conv2d_transpose_1_kernel_v_read_readvariableop=
9savev2_adam_conv2d_transpose_1_bias_v_read_readvariableop5
1savev2_adam_my_dense_kernel_v_read_readvariableop3
/savev2_adam_my_dense_bias_v_read_readvariableop
savev2_const

identity_1ИвMergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/partБ
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : У
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ═%
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:@*
dtype0*Ў$
valueь$Bщ$@B@layer_with_weights-0/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-0/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-1/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-1/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-2/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-2/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-3/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-3/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-0/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-0/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-1/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-1/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-2/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-2/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-3/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-3/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-0/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-0/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-1/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-1/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-2/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-2/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-3/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\layer_with_weights-3/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHЁ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:@*
dtype0*Х
valueЛBИ@B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B е
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0<savev2_separable_conv2d_depthwise_kernel_read_readvariableop<savev2_separable_conv2d_pointwise_kernel_read_readvariableop0savev2_separable_conv2d_bias_read_readvariableop>savev2_separable_conv2d_1_depthwise_kernel_read_readvariableop>savev2_separable_conv2d_1_pointwise_kernel_read_readvariableop2savev2_separable_conv2d_1_bias_read_readvariableop>savev2_separable_conv2d_2_depthwise_kernel_read_readvariableop>savev2_separable_conv2d_2_pointwise_kernel_read_readvariableop2savev2_separable_conv2d_2_bias_read_readvariableop>savev2_separable_conv2d_3_depthwise_kernel_read_readvariableop>savev2_separable_conv2d_3_pointwise_kernel_read_readvariableop2savev2_separable_conv2d_3_bias_read_readvariableop2savev2_conv2d_transpose_kernel_read_readvariableop0savev2_conv2d_transpose_bias_read_readvariableop4savev2_conv2d_transpose_1_kernel_read_readvariableop2savev2_conv2d_transpose_1_bias_read_readvariableop*savev2_my_dense_kernel_read_readvariableop(savev2_my_dense_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop+savev2_add_metric_total_read_readvariableop+savev2_add_metric_count_read_readvariableopCsavev2_adam_separable_conv2d_depthwise_kernel_m_read_readvariableopCsavev2_adam_separable_conv2d_pointwise_kernel_m_read_readvariableop7savev2_adam_separable_conv2d_bias_m_read_readvariableopEsavev2_adam_separable_conv2d_1_depthwise_kernel_m_read_readvariableopEsavev2_adam_separable_conv2d_1_pointwise_kernel_m_read_readvariableop9savev2_adam_separable_conv2d_1_bias_m_read_readvariableopEsavev2_adam_separable_conv2d_2_depthwise_kernel_m_read_readvariableopEsavev2_adam_separable_conv2d_2_pointwise_kernel_m_read_readvariableop9savev2_adam_separable_conv2d_2_bias_m_read_readvariableopEsavev2_adam_separable_conv2d_3_depthwise_kernel_m_read_readvariableopEsavev2_adam_separable_conv2d_3_pointwise_kernel_m_read_readvariableop9savev2_adam_separable_conv2d_3_bias_m_read_readvariableop9savev2_adam_conv2d_transpose_kernel_m_read_readvariableop7savev2_adam_conv2d_transpose_bias_m_read_readvariableop;savev2_adam_conv2d_transpose_1_kernel_m_read_readvariableop9savev2_adam_conv2d_transpose_1_bias_m_read_readvariableop1savev2_adam_my_dense_kernel_m_read_readvariableop/savev2_adam_my_dense_bias_m_read_readvariableopCsavev2_adam_separable_conv2d_depthwise_kernel_v_read_readvariableopCsavev2_adam_separable_conv2d_pointwise_kernel_v_read_readvariableop7savev2_adam_separable_conv2d_bias_v_read_readvariableopEsavev2_adam_separable_conv2d_1_depthwise_kernel_v_read_readvariableopEsavev2_adam_separable_conv2d_1_pointwise_kernel_v_read_readvariableop9savev2_adam_separable_conv2d_1_bias_v_read_readvariableopEsavev2_adam_separable_conv2d_2_depthwise_kernel_v_read_readvariableopEsavev2_adam_separable_conv2d_2_pointwise_kernel_v_read_readvariableop9savev2_adam_separable_conv2d_2_bias_v_read_readvariableopEsavev2_adam_separable_conv2d_3_depthwise_kernel_v_read_readvariableopEsavev2_adam_separable_conv2d_3_pointwise_kernel_v_read_readvariableop9savev2_adam_separable_conv2d_3_bias_v_read_readvariableop9savev2_adam_conv2d_transpose_kernel_v_read_readvariableop7savev2_adam_conv2d_transpose_bias_v_read_readvariableop;savev2_adam_conv2d_transpose_1_kernel_v_read_readvariableop9savev2_adam_conv2d_transpose_1_bias_v_read_readvariableop1savev2_adam_my_dense_kernel_v_read_readvariableop/savev2_adam_my_dense_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *N
dtypesD
B2@	Р
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:Л
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*у
_input_shapes╤
╬: ::::::::::::::::::: : : : : : : : : ::::::::::::::::::::::::::::::::::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
::,(
&
_output_shapes
:: 	

_output_shapes
::,
(
&
_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
::, (
&
_output_shapes
:: !

_output_shapes
::,"(
&
_output_shapes
::,#(
&
_output_shapes
:: $

_output_shapes
::,%(
&
_output_shapes
::,&(
&
_output_shapes
:: '

_output_shapes
::,((
&
_output_shapes
:: )

_output_shapes
::,*(
&
_output_shapes
:: +

_output_shapes
::$, 

_output_shapes

:: -

_output_shapes
::,.(
&
_output_shapes
::,/(
&
_output_shapes
:: 0

_output_shapes
::,1(
&
_output_shapes
::,2(
&
_output_shapes
:: 3

_output_shapes
::,4(
&
_output_shapes
::,5(
&
_output_shapes
:: 6

_output_shapes
::,7(
&
_output_shapes
::,8(
&
_output_shapes
:: 9

_output_shapes
::,:(
&
_output_shapes
:: ;

_output_shapes
::,<(
&
_output_shapes
:: =

_output_shapes
::$> 

_output_shapes

:: ?

_output_shapes
::@

_output_shapes
: 
┬
с
&__inference_model_1_layer_call_fn_5188

img_in

weights_in
true_labels!
unknown:#
	unknown_0:
	unknown_1:#
	unknown_2:#
	unknown_3:
	unknown_4:#
	unknown_5:#
	unknown_6:
	unknown_7:#
	unknown_8:#
	unknown_9:

unknown_10:$

unknown_11:

unknown_12:$

unknown_13:

unknown_14:

unknown_15:

unknown_16:

unknown_17: 

unknown_18: ИвStatefulPartitionedCall▐
StatefulPartitionedCallStatefulPartitionedCallimg_in
weights_intrue_labelsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18*"
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:         *4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_model_1_layer_call_and_return_conditional_losses_5100*(
_construction_contextkEagerRuntime*К
_input_shapesy
w:         └└:         └└:         └└: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:         └└
 
_user_specified_nameimg_in:YU
-
_output_shapes
:         └└
$
_user_specified_name
weights_in:ZV
-
_output_shapes
:         └└
%
_user_specified_nametrue_labels
к
Г
L__inference_separable_conv2d_3_layer_call_and_return_conditional_losses_4600

inputsB
(separable_conv2d_readvariableop_resource:D
*separable_conv2d_readvariableop_1_resource:-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвseparable_conv2d/ReadVariableOpв!separable_conv2d/ReadVariableOp_1Р
separable_conv2d/ReadVariableOpReadVariableOp(separable_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ф
!separable_conv2d/ReadVariableOp_1ReadVariableOp*separable_conv2d_readvariableop_1_resource*&
_output_shapes
:*
dtype0o
separable_conv2d/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            o
separable_conv2d/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      ╪
separable_conv2d/depthwiseDepthwiseConv2dNativeinputs'separable_conv2d/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           *
paddingSAME*
strides
▀
separable_conv2dConv2D#separable_conv2d/depthwise:output:0)separable_conv2d/ReadVariableOp_1:value:0*
T0*A
_output_shapes/
-:+                           *
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0Щ
BiasAddBiasAddseparable_conv2d:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           j
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+                           {
IdentityIdentityRelu:activations:0^NoOp*
T0*A
_output_shapes/
-:+                           е
NoOpNoOp^BiasAdd/ReadVariableOp ^separable_conv2d/ReadVariableOp"^separable_conv2d/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*F
_input_shapes5
3:+                           : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
separable_conv2d/ReadVariableOpseparable_conv2d/ReadVariableOp2F
!separable_conv2d/ReadVariableOp_1!separable_conv2d/ReadVariableOp_1:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
ч
n
B__inference_add_loss_layer_call_and_return_conditional_losses_6163

inputs
identity

identity_1J
IdentityIdentityinputs*
T0*#
_output_shapes
:         L

Identity_1Identityinputs*
T0*#
_output_shapes
:         "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*"
_input_shapes
:         :K G
#
_output_shapes
:         
 
_user_specified_nameinputs
Н
]
A__inference_reshape_layer_call_and_return_conditional_losses_6152

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╤
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskR
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :└R
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :└П
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:j
ReshapeReshapeinputsReshape/shape:output:0*
T0*-
_output_shapes
:         └└^
IdentityIdentityReshape:output:0*
T0*-
_output_shapes
:         └└"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+                           :i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs"┐L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Й
serving_defaultї
C
img_in9
serving_default_img_in:0         └└
I
true_labels:
serving_default_true_labels:0         └└
G

weights_in9
serving_default_weights_in:0         └└tensorflow/serving/predict:╘┐
У
layer-0
layer-1
layer-2
layer_with_weights-0
layer-3
layer_with_weights-1
layer-4
layer-5
layer_with_weights-2
layer-6
layer_with_weights-3
layer-7
	layer-8

layer_with_weights-4

layer-9
layer_with_weights-5
layer-10
layer_with_weights-6
layer-11
layer-12
layer-13
layer-14
layer-15
layer-16
layer-17
layer-18
layer-19
layer-20
layer-21
layer-22
layer-23
layer-24
layer-25
layer-26
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses
"_default_save_signature
#	optimizer
$loss
%
signatures"
_tf_keras_network
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
¤
&	variables
'trainable_variables
(regularization_losses
)	keras_api
*__call__
*+&call_and_return_all_conditional_losses
,depthwise_kernel
-pointwise_kernel
.bias
 /_jit_compiled_convolution_op"
_tf_keras_layer
¤
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses
6depthwise_kernel
7pointwise_kernel
8bias
 9_jit_compiled_convolution_op"
_tf_keras_layer
е
:	variables
;trainable_variables
<regularization_losses
=	keras_api
>__call__
*?&call_and_return_all_conditional_losses"
_tf_keras_layer
¤
@	variables
Atrainable_variables
Bregularization_losses
C	keras_api
D__call__
*E&call_and_return_all_conditional_losses
Fdepthwise_kernel
Gpointwise_kernel
Hbias
 I_jit_compiled_convolution_op"
_tf_keras_layer
¤
J	variables
Ktrainable_variables
Lregularization_losses
M	keras_api
N__call__
*O&call_and_return_all_conditional_losses
Pdepthwise_kernel
Qpointwise_kernel
Rbias
 S_jit_compiled_convolution_op"
_tf_keras_layer
е
T	variables
Utrainable_variables
Vregularization_losses
W	keras_api
X__call__
*Y&call_and_return_all_conditional_losses"
_tf_keras_layer
▌
Z	variables
[trainable_variables
\regularization_losses
]	keras_api
^__call__
*_&call_and_return_all_conditional_losses

`kernel
abias
 b_jit_compiled_convolution_op"
_tf_keras_layer
▌
c	variables
dtrainable_variables
eregularization_losses
f	keras_api
g__call__
*h&call_and_return_all_conditional_losses

ikernel
jbias
 k_jit_compiled_convolution_op"
_tf_keras_layer
╗
l	variables
mtrainable_variables
nregularization_losses
o	keras_api
p__call__
*q&call_and_return_all_conditional_losses

rkernel
sbias"
_tf_keras_layer
е
t	variables
utrainable_variables
vregularization_losses
w	keras_api
x__call__
*y&call_and_return_all_conditional_losses"
_tf_keras_layer
(
z	keras_api"
_tf_keras_layer
(
{	keras_api"
_tf_keras_layer
(
|	keras_api"
_tf_keras_layer
(
}	keras_api"
_tf_keras_layer
(
~	keras_api"
_tf_keras_layer
(
	keras_api"
_tf_keras_layer
л
А	variables
Бtrainable_variables
Вregularization_losses
Г	keras_api
Д__call__
+Е&call_and_return_all_conditional_losses"
_tf_keras_layer
)
Ж	keras_api"
_tf_keras_layer
)
З	keras_api"
_tf_keras_layer
)
И	keras_api"
_tf_keras_layer
)
Й	keras_api"
_tf_keras_layer
)
К	keras_api"
_tf_keras_layer
)
Л	keras_api"
_tf_keras_layer
л
М	variables
Нtrainable_variables
Оregularization_losses
П	keras_api
Р__call__
+С&call_and_return_all_conditional_losses"
_tf_keras_layer
ж
,0
-1
.2
63
74
85
F6
G7
H8
P9
Q10
R11
`12
a13
i14
j15
r16
s17"
trackable_list_wrapper
ж
,0
-1
.2
63
74
85
F6
G7
H8
P9
Q10
R11
`12
a13
i14
j15
r16
s17"
trackable_list_wrapper
 "
trackable_list_wrapper
╧
Тnon_trainable_variables
Уlayers
Фmetrics
 Хlayer_regularization_losses
Цlayer_metrics
	variables
trainable_variables
regularization_losses
 __call__
"_default_save_signature
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses"
_generic_user_object
╓
Чtrace_0
Шtrace_1
Щtrace_2
Ъtrace_32у
&__inference_model_1_layer_call_fn_4925
&__inference_model_1_layer_call_fn_5471
&__inference_model_1_layer_call_fn_5517
&__inference_model_1_layer_call_fn_5188└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 zЧtrace_0zШtrace_1zЩtrace_2zЪtrace_3
┬
Ыtrace_0
Ьtrace_1
Эtrace_2
Юtrace_32╧
A__inference_model_1_layer_call_and_return_conditional_losses_5695
A__inference_model_1_layer_call_and_return_conditional_losses_5873
A__inference_model_1_layer_call_and_return_conditional_losses_5280
A__inference_model_1_layer_call_and_return_conditional_losses_5372└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 zЫtrace_0zЬtrace_1zЭtrace_2zЮtrace_3
тB▀
__inference__wrapped_model_4481img_in
weights_intrue_labels"Ш
С▓Н
FullArgSpec
argsЪ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
└
	Яiter
аbeta_1
бbeta_2

вdecay
гlearning_rate,mГ-mД.mЕ6mЖ7mЗ8mИFmЙGmКHmЛPmМQmНRmО`mПamРimСjmТrmУsmФ,vХ-vЦ.vЧ6vШ7vЩ8vЪFvЫGvЬHvЭPvЮQvЯRvа`vбavвivгjvдrvеsvж"
	optimizer
 "
trackable_dict_wrapper
-
дserving_default"
signature_map
5
,0
-1
.2"
trackable_list_wrapper
5
,0
-1
.2"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
еnon_trainable_variables
жlayers
зmetrics
 иlayer_regularization_losses
йlayer_metrics
&	variables
'trainable_variables
(regularization_losses
*__call__
*+&call_and_return_all_conditional_losses
&+"call_and_return_conditional_losses"
_generic_user_object
ї
кtrace_02╓
/__inference_separable_conv2d_layer_call_fn_5884в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zкtrace_0
Р
лtrace_02ё
J__inference_separable_conv2d_layer_call_and_return_conditional_losses_5900в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zлtrace_0
;:92!separable_conv2d/depthwise_kernel
;:92!separable_conv2d/pointwise_kernel
#:!2separable_conv2d/bias
┤2▒о
г▓Я
FullArgSpec'
argsЪ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 0
5
60
71
82"
trackable_list_wrapper
5
60
71
82"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
мnon_trainable_variables
нlayers
оmetrics
 пlayer_regularization_losses
░layer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses"
_generic_user_object
ў
▒trace_02╪
1__inference_separable_conv2d_1_layer_call_fn_5911в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z▒trace_0
Т
▓trace_02є
L__inference_separable_conv2d_1_layer_call_and_return_conditional_losses_5927в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z▓trace_0
=:;2#separable_conv2d_1/depthwise_kernel
=:;2#separable_conv2d_1/pointwise_kernel
%:#2separable_conv2d_1/bias
┤2▒о
г▓Я
FullArgSpec'
argsЪ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
│non_trainable_variables
┤layers
╡metrics
 ╢layer_regularization_losses
╖layer_metrics
:	variables
;trainable_variables
<regularization_losses
>__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses"
_generic_user_object
Є
╕trace_02╙
,__inference_max_pooling2d_layer_call_fn_5932в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╕trace_0
Н
╣trace_02ю
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_5937в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╣trace_0
5
F0
G1
H2"
trackable_list_wrapper
5
F0
G1
H2"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
║non_trainable_variables
╗layers
╝metrics
 ╜layer_regularization_losses
╛layer_metrics
@	variables
Atrainable_variables
Bregularization_losses
D__call__
*E&call_and_return_all_conditional_losses
&E"call_and_return_conditional_losses"
_generic_user_object
ў
┐trace_02╪
1__inference_separable_conv2d_2_layer_call_fn_5948в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z┐trace_0
Т
└trace_02є
L__inference_separable_conv2d_2_layer_call_and_return_conditional_losses_5964в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z└trace_0
=:;2#separable_conv2d_2/depthwise_kernel
=:;2#separable_conv2d_2/pointwise_kernel
%:#2separable_conv2d_2/bias
┤2▒о
г▓Я
FullArgSpec'
argsЪ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 0
5
P0
Q1
R2"
trackable_list_wrapper
5
P0
Q1
R2"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
┴non_trainable_variables
┬layers
├metrics
 ─layer_regularization_losses
┼layer_metrics
J	variables
Ktrainable_variables
Lregularization_losses
N__call__
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses"
_generic_user_object
ў
╞trace_02╪
1__inference_separable_conv2d_3_layer_call_fn_5975в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╞trace_0
Т
╟trace_02є
L__inference_separable_conv2d_3_layer_call_and_return_conditional_losses_5991в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╟trace_0
=:;2#separable_conv2d_3/depthwise_kernel
=:;2#separable_conv2d_3/pointwise_kernel
%:#2separable_conv2d_3/bias
┤2▒о
г▓Я
FullArgSpec'
argsЪ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
╚non_trainable_variables
╔layers
╩metrics
 ╦layer_regularization_losses
╠layer_metrics
T	variables
Utrainable_variables
Vregularization_losses
X__call__
*Y&call_and_return_all_conditional_losses
&Y"call_and_return_conditional_losses"
_generic_user_object
Є
═trace_02╙
,__inference_up_sampling2d_layer_call_fn_5996в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z═trace_0
Н
╬trace_02ю
G__inference_up_sampling2d_layer_call_and_return_conditional_losses_6008в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╬trace_0
.
`0
a1"
trackable_list_wrapper
.
`0
a1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
╧non_trainable_variables
╨layers
╤metrics
 ╥layer_regularization_losses
╙layer_metrics
Z	variables
[trainable_variables
\regularization_losses
^__call__
*_&call_and_return_all_conditional_losses
&_"call_and_return_conditional_losses"
_generic_user_object
ї
╘trace_02╓
/__inference_conv2d_transpose_layer_call_fn_6017в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╘trace_0
Р
╒trace_02ё
J__inference_conv2d_transpose_layer_call_and_return_conditional_losses_6051в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╒trace_0
1:/2conv2d_transpose/kernel
#:!2conv2d_transpose/bias
┤2▒о
г▓Я
FullArgSpec'
argsЪ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 0
.
i0
j1"
trackable_list_wrapper
.
i0
j1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
╓non_trainable_variables
╫layers
╪metrics
 ┘layer_regularization_losses
┌layer_metrics
c	variables
dtrainable_variables
eregularization_losses
g__call__
*h&call_and_return_all_conditional_losses
&h"call_and_return_conditional_losses"
_generic_user_object
ў
█trace_02╪
1__inference_conv2d_transpose_1_layer_call_fn_6060в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z█trace_0
Т
▄trace_02є
L__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_6094в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z▄trace_0
3:12conv2d_transpose_1/kernel
%:#2conv2d_transpose_1/bias
┤2▒о
г▓Я
FullArgSpec'
argsЪ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 0
.
r0
s1"
trackable_list_wrapper
.
r0
s1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
▌non_trainable_variables
▐layers
▀metrics
 рlayer_regularization_losses
сlayer_metrics
l	variables
mtrainable_variables
nregularization_losses
p__call__
*q&call_and_return_all_conditional_losses
&q"call_and_return_conditional_losses"
_generic_user_object
э
тtrace_02╬
'__inference_my_dense_layer_call_fn_6103в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zтtrace_0
И
уtrace_02щ
B__inference_my_dense_layer_call_and_return_conditional_losses_6134в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zуtrace_0
!:2my_dense/kernel
:2my_dense/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
фnon_trainable_variables
хlayers
цmetrics
 чlayer_regularization_losses
шlayer_metrics
t	variables
utrainable_variables
vregularization_losses
x__call__
*y&call_and_return_all_conditional_losses
&y"call_and_return_conditional_losses"
_generic_user_object
ь
щtrace_02═
&__inference_reshape_layer_call_fn_6139в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zщtrace_0
З
ъtrace_02ш
A__inference_reshape_layer_call_and_return_conditional_losses_6152в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zъtrace_0
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
ыnon_trainable_variables
ьlayers
эmetrics
 юlayer_regularization_losses
яlayer_metrics
А	variables
Бtrainable_variables
Вregularization_losses
Д__call__
+Е&call_and_return_all_conditional_losses
'Е"call_and_return_conditional_losses"
_generic_user_object
э
Ёtrace_02╬
'__inference_add_loss_layer_call_fn_6158в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zЁtrace_0
И
ёtrace_02щ
B__inference_add_loss_layer_call_and_return_conditional_losses_6163в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zёtrace_0
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
Єnon_trainable_variables
єlayers
Їmetrics
 їlayer_regularization_losses
Ўlayer_metrics
М	variables
Нtrainable_variables
Оregularization_losses
Р__call__
+С&call_and_return_all_conditional_losses
'С"call_and_return_conditional_losses"
_generic_user_object
я
ўtrace_02╨
)__inference_add_metric_layer_call_fn_6172в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zўtrace_0
К
°trace_02ы
D__inference_add_metric_layer_call_and_return_conditional_losses_6186в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z°trace_0
 "
trackable_list_wrapper
ю
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26"
trackable_list_wrapper
0
∙0
·1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
СBО
&__inference_model_1_layer_call_fn_4925img_in
weights_intrue_labels"└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
ОBЛ
&__inference_model_1_layer_call_fn_5471inputs/0inputs/1inputs/2"└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
ОBЛ
&__inference_model_1_layer_call_fn_5517inputs/0inputs/1inputs/2"└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
СBО
&__inference_model_1_layer_call_fn_5188img_in
weights_intrue_labels"└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
йBж
A__inference_model_1_layer_call_and_return_conditional_losses_5695inputs/0inputs/1inputs/2"└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
йBж
A__inference_model_1_layer_call_and_return_conditional_losses_5873inputs/0inputs/1inputs/2"└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
мBй
A__inference_model_1_layer_call_and_return_conditional_losses_5280img_in
weights_intrue_labels"└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
мBй
A__inference_model_1_layer_call_and_return_conditional_losses_5372img_in
weights_intrue_labels"└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
▀B▄
"__inference_signature_wrapper_5425img_intrue_labels
weights_in"Ф
Н▓Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
уBр
/__inference_separable_conv2d_layer_call_fn_5884inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
■B√
J__inference_separable_conv2d_layer_call_and_return_conditional_losses_5900inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
хBт
1__inference_separable_conv2d_1_layer_call_fn_5911inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
АB¤
L__inference_separable_conv2d_1_layer_call_and_return_conditional_losses_5927inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
рB▌
,__inference_max_pooling2d_layer_call_fn_5932inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
√B°
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_5937inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
хBт
1__inference_separable_conv2d_2_layer_call_fn_5948inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
АB¤
L__inference_separable_conv2d_2_layer_call_and_return_conditional_losses_5964inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
хBт
1__inference_separable_conv2d_3_layer_call_fn_5975inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
АB¤
L__inference_separable_conv2d_3_layer_call_and_return_conditional_losses_5991inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
рB▌
,__inference_up_sampling2d_layer_call_fn_5996inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
√B°
G__inference_up_sampling2d_layer_call_and_return_conditional_losses_6008inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
уBр
/__inference_conv2d_transpose_layer_call_fn_6017inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
■B√
J__inference_conv2d_transpose_layer_call_and_return_conditional_losses_6051inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
хBт
1__inference_conv2d_transpose_1_layer_call_fn_6060inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
АB¤
L__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_6094inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
█B╪
'__inference_my_dense_layer_call_fn_6103inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЎBє
B__inference_my_dense_layer_call_and_return_conditional_losses_6134inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
┌B╫
&__inference_reshape_layer_call_fn_6139inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
їBЄ
A__inference_reshape_layer_call_and_return_conditional_losses_6152inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
█B╪
'__inference_add_loss_layer_call_fn_6158inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЎBє
B__inference_add_loss_layer_call_and_return_conditional_losses_6163inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
·0"
trackable_list_wrapper
 "
trackable_list_wrapper
8
·weighted_accuracy"
trackable_dict_wrapper
▌B┌
)__inference_add_metric_layer_call_fn_6172inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
°Bї
D__inference_add_metric_layer_call_and_return_conditional_losses_6186inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
R
√	variables
№	keras_api

¤total

■count"
_tf_keras_metric
R
 	variables
А	keras_api

Бtotal

Вcount"
_tf_keras_metric
0
¤0
■1"
trackable_list_wrapper
.
√	variables"
_generic_user_object
:  (2total
:  (2count
0
Б0
В1"
trackable_list_wrapper
.
 	variables"
_generic_user_object
:  (2add_metric/total
:  (2add_metric/count
@:>2(Adam/separable_conv2d/depthwise_kernel/m
@:>2(Adam/separable_conv2d/pointwise_kernel/m
(:&2Adam/separable_conv2d/bias/m
B:@2*Adam/separable_conv2d_1/depthwise_kernel/m
B:@2*Adam/separable_conv2d_1/pointwise_kernel/m
*:(2Adam/separable_conv2d_1/bias/m
B:@2*Adam/separable_conv2d_2/depthwise_kernel/m
B:@2*Adam/separable_conv2d_2/pointwise_kernel/m
*:(2Adam/separable_conv2d_2/bias/m
B:@2*Adam/separable_conv2d_3/depthwise_kernel/m
B:@2*Adam/separable_conv2d_3/pointwise_kernel/m
*:(2Adam/separable_conv2d_3/bias/m
6:42Adam/conv2d_transpose/kernel/m
(:&2Adam/conv2d_transpose/bias/m
8:62 Adam/conv2d_transpose_1/kernel/m
*:(2Adam/conv2d_transpose_1/bias/m
&:$2Adam/my_dense/kernel/m
 :2Adam/my_dense/bias/m
@:>2(Adam/separable_conv2d/depthwise_kernel/v
@:>2(Adam/separable_conv2d/pointwise_kernel/v
(:&2Adam/separable_conv2d/bias/v
B:@2*Adam/separable_conv2d_1/depthwise_kernel/v
B:@2*Adam/separable_conv2d_1/pointwise_kernel/v
*:(2Adam/separable_conv2d_1/bias/v
B:@2*Adam/separable_conv2d_2/depthwise_kernel/v
B:@2*Adam/separable_conv2d_2/pointwise_kernel/v
*:(2Adam/separable_conv2d_2/bias/v
B:@2*Adam/separable_conv2d_3/depthwise_kernel/v
B:@2*Adam/separable_conv2d_3/pointwise_kernel/v
*:(2Adam/separable_conv2d_3/bias/v
6:42Adam/conv2d_transpose/kernel/v
(:&2Adam/conv2d_transpose/bias/v
8:62 Adam/conv2d_transpose_1/kernel/v
*:(2Adam/conv2d_transpose_1/bias/v
&:$2Adam/my_dense/kernel/v
 :2Adam/my_dense/bias/vр
__inference__wrapped_model_4481╝,-.678FGHPQR`aijrsБВЬвШ
РвМ
ЙЪЕ
*К'
img_in         └└
*К'

weights_in         └└
+К(
true_labels         └└
к "к ▒
B__inference_add_loss_layer_call_and_return_conditional_losses_6163k+в(
!в
К
inputs         
к "<в9
К
0         
Ъ
К
1/0         n
'__inference_add_loss_layer_call_fn_6158C+в(
!в
К
inputs         
к "К         Ю
D__inference_add_metric_layer_call_and_return_conditional_losses_6186VБВ+в(
!в
К
inputs         
к "!в
К
0         
Ъ v
)__inference_add_metric_layer_call_fn_6172IБВ+в(
!в
К
inputs         
к "К         с
L__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_6094РijIвF
?в<
:К7
inputs+                           
к "?в<
5К2
0+                           
Ъ ╣
1__inference_conv2d_transpose_1_layer_call_fn_6060ГijIвF
?в<
:К7
inputs+                           
к "2К/+                           ▀
J__inference_conv2d_transpose_layer_call_and_return_conditional_losses_6051Р`aIвF
?в<
:К7
inputs+                           
к "?в<
5К2
0+                           
Ъ ╖
/__inference_conv2d_transpose_layer_call_fn_6017Г`aIвF
?в<
:К7
inputs+                           
к "2К/+                           ъ
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_5937ЮRвO
HвE
CК@
inputs4                                    
к "HвE
>К;
04                                    
Ъ ┬
,__inference_max_pooling2d_layer_call_fn_5932СRвO
HвE
CК@
inputs4                                    
к ";К84                                    п
A__inference_model_1_layer_call_and_return_conditional_losses_5280щ,-.678FGHPQR`aijrsБВдва
ШвФ
ЙЪЕ
*К'
img_in         └└
*К'

weights_in         └└
+К(
true_labels         └└
p 

 
к "(в%
Ъ 
Ъ
К
1/0         п
A__inference_model_1_layer_call_and_return_conditional_losses_5372щ,-.678FGHPQR`aijrsБВдва
ШвФ
ЙЪЕ
*К'
img_in         └└
*К'

weights_in         └└
+К(
true_labels         └└
p

 
к "(в%
Ъ 
Ъ
К
1/0         м
A__inference_model_1_layer_call_and_return_conditional_losses_5695ц,-.678FGHPQR`aijrsБВбвЭ
ХвС
ЖЪВ
,К)
inputs/0         └└
(К%
inputs/1         └└
(К%
inputs/2         └└
p 

 
к "(в%
Ъ 
Ъ
К
1/0         м
A__inference_model_1_layer_call_and_return_conditional_losses_5873ц,-.678FGHPQR`aijrsБВбвЭ
ХвС
ЖЪВ
,К)
inputs/0         └└
(К%
inputs/1         └└
(К%
inputs/2         └└
p

 
к "(в%
Ъ 
Ъ
К
1/0         я
&__inference_model_1_layer_call_fn_4925─,-.678FGHPQR`aijrsБВдва
ШвФ
ЙЪЕ
*К'
img_in         └└
*К'

weights_in         └└
+К(
true_labels         └└
p 

 
к "Ъ я
&__inference_model_1_layer_call_fn_5188─,-.678FGHPQR`aijrsБВдва
ШвФ
ЙЪЕ
*К'
img_in         └└
*К'

weights_in         └└
+К(
true_labels         └└
p

 
к "Ъ ь
&__inference_model_1_layer_call_fn_5471┴,-.678FGHPQR`aijrsБВбвЭ
ХвС
ЖЪВ
,К)
inputs/0         └└
(К%
inputs/1         └└
(К%
inputs/2         └└
p 

 
к "Ъ ь
&__inference_model_1_layer_call_fn_5517┴,-.678FGHPQR`aijrsБВбвЭ
ХвС
ЖЪВ
,К)
inputs/0         └└
(К%
inputs/1         └└
(К%
inputs/2         └└
p

 
к "Ъ ╫
B__inference_my_dense_layer_call_and_return_conditional_losses_6134РrsIвF
?в<
:К7
inputs+                           
к "?в<
5К2
0+                           
Ъ п
'__inference_my_dense_layer_call_fn_6103ГrsIвF
?в<
:К7
inputs+                           
к "2К/+                           ╜
A__inference_reshape_layer_call_and_return_conditional_losses_6152xIвF
?в<
:К7
inputs+                           
к "+в(
!К
0         └└
Ъ Х
&__inference_reshape_layer_call_fn_6139kIвF
?в<
:К7
inputs+                           
к "К         └└т
L__inference_separable_conv2d_1_layer_call_and_return_conditional_losses_5927С678IвF
?в<
:К7
inputs+                           
к "?в<
5К2
0+                           
Ъ ║
1__inference_separable_conv2d_1_layer_call_fn_5911Д678IвF
?в<
:К7
inputs+                           
к "2К/+                           т
L__inference_separable_conv2d_2_layer_call_and_return_conditional_losses_5964СFGHIвF
?в<
:К7
inputs+                           
к "?в<
5К2
0+                           
Ъ ║
1__inference_separable_conv2d_2_layer_call_fn_5948ДFGHIвF
?в<
:К7
inputs+                           
к "2К/+                           т
L__inference_separable_conv2d_3_layer_call_and_return_conditional_losses_5991СPQRIвF
?в<
:К7
inputs+                           
к "?в<
5К2
0+                           
Ъ ║
1__inference_separable_conv2d_3_layer_call_fn_5975ДPQRIвF
?в<
:К7
inputs+                           
к "2К/+                           р
J__inference_separable_conv2d_layer_call_and_return_conditional_losses_5900С,-.IвF
?в<
:К7
inputs+                           
к "?в<
5К2
0+                           
Ъ ╕
/__inference_separable_conv2d_layer_call_fn_5884Д,-.IвF
?в<
:К7
inputs+                           
к "2К/+                           Г
"__inference_signature_wrapper_5425▄,-.678FGHPQR`aijrsБВ╝в╕
в 
░км
4
img_in*К'
img_in         └└
:
true_labels+К(
true_labels         └└
8

weights_in*К'

weights_in         └└"к ъ
G__inference_up_sampling2d_layer_call_and_return_conditional_losses_6008ЮRвO
HвE
CК@
inputs4                                    
к "HвE
>К;
04                                    
Ъ ┬
,__inference_up_sampling2d_layer_call_fn_5996СRвO
HвE
CК@
inputs4                                    
к ";К84                                    