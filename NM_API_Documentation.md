`#include "myNM.h"`

## Non-Linear System Solver

### Bisection()

---



finds an approximation for the root of real-valued function, f(x)=-0

 `double bisection(double _a, double _b, double _tol);`

### Parameters

* _a : initial values of x, left side of interval

* _b : FInal values of x, right side of interval

* _tol : tolerance that stops iteration when it is within given error interval

  

  ## Example code

  ```c
  	double tol = 0.0000001;
  	double x0 = 1.5;
  	double deflectionresult_newton;
  	double a = 0.0;
  	double b = 4.0;
  	double deflectionresult_bisection;
  	double max_deflection;
  	double max_deflectionbisection;
  
  	deflectionresult_bisection = bisection(a, b,tol);
  
  ```

###  NewtonRaphson()

----



finds an approximation for the root of real-valued function, f(x). It uses initial value of x. However, inappropriate selection of initial value will make the solution diverge.

`double newtonraphson(double _x0, double _tol)`



#### Parameters

* _x0 : Initial approximation of x

* _tol : tolerance that stops iteration when it is within given error interval

  #### Example Codes

  ```c
  	double tol = 0.0000001;
  	double x0 = 1.5;
  	deflectionresult_newton = newtonraphson(x0, tol);
  ```

### Secant() 

----



finds an approximation for the root of real-valued function, f(x)=-0. Secant method complements Newton Raphson's method, which has shortcomings of divergence, by using slope of the two points of the given function

`double secant(double _x, double _x1, double _tol)`

### Parameters

* _x : initial guess of x

* _x1 : second guess of x

* _tol : _tol : tolerance that stops iteration when it is within given error interval

  #### Example Codes

```c
	double tol = 0.0000001;
	double x0 = 1.5;
	double x1 = 2.5
	deflectionresult_secant = newtonraphson(x0,x1, tol);
```





----



# Linear System Solver

### addMat()

---



This function serves as addition of two matrices. Two matrices should have equal size of rows and columns 

`Matrix	addMat(Matrix _A, Matrix _B)`

#### Parameters

* Matrix _A : Matrix A, should be same size with Matrix B

* Matrix _B : Matrix B, should be same size with Matrix A

  ### Example Codes

  ```c
  Matrix matA = txt2Mat(path, "prob1_matA");
  Matrix matb = txt2Mat(path, "prob1_matb");
  addMat(matA,matb);
  ```

### multimat()

----



This function functions as multiplication of two matrix, and gives resultant value to the input. This function complements some crucial shortcomings. When calculation function such as addMat, multimat is implemented to another function, return value of function is not changed due to input function parameters' re-definition which is caused by same-address usage in array. In this function, by giving a resultant matrix as a input parameter, same-address usage is not occurred in function.

`void multimat(Matrix _A, Matrix _b, Matrix _multi)`

#### Parameters

* Matrix _A : Input matrix A, that has size of mxn

* Matrix _b : input matrix b, that has size of nxm

* Matrix _multi : output matrix, that has size of mxm

  #### Example Codes

  ```c
  Matrix _A = txt2Mat(path, "prob1_matA");
  Matrix matb = txt2Mat(path, "prob1_matb");
  Matrix Out = createMat(_A.rows,b.cols);
  multimat(_A,matb,Out);
  ```

### backSub()

---



This function serves as finding appropriate x for the Linear system Ax=b, by using back-substitution method.

#### Parameters

* Matrix _A : Input Matrix A . should be (nxn) square

* Matrix _b : Input Matrix b . should be (nx1)

  #### Example Codes

  ```c
  	Matrix matA = txt2Mat(path, "prob1_matA");
  	Matrix vecb = txt2Mat(path, "prob1_vecb");
      backSub(matA, vecb);
  ```

### transpose()

---

This function serves as giving transpose of input matrix A.

`void transpose(Matrix _A, Matrix _B)`

#### Parameters

* Matrix _A : Input Matrix of A that is to be transposed, should be size of (nxn)
* Matrix _B : Transposed Matrix B, size of (nxn)

#### Example Codes

```c
Matrix matA = txt2Mat(path, "prob1_matA");
Matrix vecb = createMat(A.rows,A.cols);
transpose(matA,vecb);
```



### GaussElim()

---



This function serves as giving Row-Echelon form of given linear system Ax=b

`void GaussElim(Matrix _A, Matrix _b)`

#### Parameters

* Matrix _A : Input Matrix A. should be (nxn) square

* Matrix _b : Input Matrix b. should be (nx1)

  #### Example Codes

  ```c
  Matrix matA = txt2Mat(path, "prob1_matA");
  Matrix vecb = txt2Mat(path, "prob1_vecb");
  GaussElim(matA,vecb);
  ```

### GausJordan()

---



This function gives Reduced Row Echelon form of given linear system Ax=b

`void GaussJordan(Matrix _A, Matrix _b)`

#### Parameters 

* Matrix _A : Input Matrix A, should be (nxn) square

* Matrix _b : Input Matrix b, should be (nx1) 

  #### Example codes

  ```c
  Matrix matA = txt2Mat(path, "prob1_matA");
  Matrix vecb = txt2Mat(path, "prob1_vecb");
  GaussJordan(matA,vecb);
  ```

### LUwithpivot()

----



This function serves as LU decomposition for the given linear system Ax=b. Also, by using permutation matrix, it gives LU decomposition with pivoting.

Matrix LUwithpivot(Matrix _A, Matrix _b, Matrix _L, Matrix _permut)

#### Parameters

* Matrix _A : Input Matrix A, should be (nxn) 

* Matrix _b : Input Matrix b, should be (nx1) 

* Matrix _L : Lower Triangle Matrix, should be (nxn) and zeros at initial condition

* Matrix _permut : Initial permutation matrix which has size of (nxn) and is zeros at initial condition

  #### Example codes

  ```c
  Matrix matA = txt2Mat(path, "prob1_matA");
  	Matrix vecb = txt2Mat(path, "prob1_vecb");
  	Matrix matTemp = createMat(matA.cols, vecb.cols);//x 생성
  	initMat(matTemp, 0);
  	Matrix L = createMat(matA.rows, matA.cols);
  	initMat(L, 0);
  	Matrix permut = createMat(matA.rows, matA.cols);
  	initMat(permut, 0);
  Matrix LU = Matrix(matA,vecb,L,permut)
  ```

### Jacobi_Iterative() 

--------



This function serves as finding approximate root of linear system Ax=b by iteration.

`void  JacobiIterative(Matrix _A, Matrix _b)`

#### Parameters

* Matrix _A : input Matrix A, should be (nxn) size

* Matrix _b : input Matrix b, should be (nx1) size

  #### Example Codes

  ```c
  	Matrix matA = txt2Mat(path, "prob1_matA");
  	Matrix vecb = txt2Mat(path, "prob1_vecb");
  	JacobiIterative(matA,vecb);
  
  ```

### Gauss-Seidal()

------

This function serves as finding approximate root of linear system Ax=b by iteration. This function is similar to Jacobi-iterative. However, as this function uses previous approximation of x, it usually converges faster than Jacobi-iterative method.

`void GaussSeidel(Matrix _A, Matrix _b)`

#### Parameters

* Matrix _A : input Matrix A, should be (nxn) size

* Matrix _b : input Matrix b, should be (nx1) size

  #### Example Codes

  ```c
  Matrix matA = txt2Mat(path, "prob1_matA");
  	Matrix vecb = txt2Mat(path, "prob1_vecb");
  	GaussSeidal(matA,vecb);
  ```

### Norm()

---



This function functions as finding vector norm of (nx1) size of Matrix

`double Norm(Matrix _b)`

#### Parameters

* Matrix _b : Input Matrix b, should be (nx1) size of Matrix

#### Example Codes

```c
Matrix vecb = txt2Mat(path, "prob1_vecb");
double norm=0;
norm = Norm(vecb)
```

### aNorm()

---

This function serves as finding L1-Norm by using colum addition. Matrix A should be (nxn) 

`double aNorm(Matrix _A)`

#### Parameters

* Matrix _A : input matrix A with size of (nxn)

#### Example Codes

```c
Matrix matA = txt2Mat(path, "prob1_matA");
double anorm=0;
anorm = aNorm(matA)
```

### infNorm()

---

This function serves as finding infinity Norm by using row addition.

`double infNorm(Matrix _A)`

#### Parameters

* Matrix _A : input Matrix with size of (nxn)

#### Example codes

```c
Matrix matA = txt2Mat(path, "prob1_matA");
double infnorm=0;
infnorm = infNorm(matA)
```

### EuclidNorm()

---



This function serves as finding Euclidean Norm by using square root(distance)

`double EuclidNorm(Matrix _A)`

#### Parameters

* Matrix _A : input Matrix A with size of (nxn)

#### Example Codes

```c
Matrix matA = txt2Mat(path, "prob1_matA");
double Eucnorm=0;
Eucnorm = EuclidNorm(matA)
```

### SolveLU

---



This function serves as finding root of LU decomposition. Input Matrix should be lu decomposed, and back, forward substitution is used to solve Ly=b, then Ux = y. 

`void SolveLU(Matrix _Low, Matrix _Upper, Matrix _Per, Matrix _vecb)`

#### Parameters

* Matrix _Low : decomposed Matrix A to the lower triangle matrix
* Matrix _Upper : decomposed Matrix A to the upper triangle matrix
* Matrix _Per : permutation matrix that is calculated in LU decomposition with pivoiting
* Matrix _vecb : Input matrix b with size of (nx1)

Obviously, input parameters Low,Upper,Per should have size of (nxn).

#### Example Codes

```c
	Matrix matA = txt2Mat(path, "prob1_matA");
	Matrix vecb = txt2Mat(path, "prob1_vecb");
	Matrix matTemp = createMat(matA.cols, vecb.cols);//x 생성
	initMat(matTemp, 0);
	Matrix L = createMat(matA.rows, matA.cols);
	initMat(L, 0);
	Matrix permut = createMat(matA.rows, matA.cols);
	initMat(permut, 0);
	permut = diagmat(matA);
	Matrix LU = Matrix(matA,vecb,L,permut);
    SolveLU(L,LU,permut,vecb);
```



### Inv()

----

This function serves as finding inverse matrix of given matrix A. In this function, Gauss Jordan elimination is used.

`Matrix inv(Matrix _A)`

#### Parameters

* Matrix _A : input matrix A which has size of (nxn)
* It returns inverse matrix of A ,size of(nxn)

#### Example Codes

```c
Matrix matA = txt2Mat(path, "prob1_matA");
Matrix inv = inv(matA);
```

### inverse2()

---

This function serves as finding inverse matrix of given matrix A by using LU decomposition

`void   inverse2(Matrix _A, Matrix _invA)`

#### Parameters

* Matrix _A : Given Matrix A, should be (nxn) size
* Matrix _invA : Storing inverse matrix values, which will be complete inverse matrix at the end of the function

#### Example Codes

```c
Matrix A = txtMat(path,"prob1_matA");
Matrix Ainv = createMat(A.rows,A.cols);
inverse2(A,Ainv);
```



### QRFact()

---

This function serves as QR factorization of given Matrix A. This function implements house hold matrix to find QR factorization. This is commonly used to find eigenvalues and eigenvectors.

`void QRFact(Matrix _A)`

#### Parameters

* Matrix _A : input Matrix A with size of (nxn)
* In this function, we can notice that Q and R matrix declared in the function is changed while the for loop ends. When for loop is terminated,  eigenvalues can be found at the diagonal components of Matrix R

#### Example Codes

```c
Matrix matA = txt2Mat(path, "prob1_matA");
QRFact(matA);
```

### lamda()

---



In this function, eigenvalues can be found by iterating QR factorization. Ultimately, eigenvalues, eigenvectors, and condition number for given matrix can be calculated.

`Matrix lamda(Matrix _A)`

#### Parameters

* Matrix _A : Input Matrix A with size of (nxn)
* returns eigenvalues of given Matrix A in matrix form, size of (nx1)

####  Example Codes 

```c
Matrix matA = txt2Mat(path, "prob1_matA");
Matrix lam = createMat(A.rows,1);
lam = lamda(matA);
```

### Cond() 

---



in this function, by giving input matrix A, it is possible to calculate its condition number characterized by singular value.

`double cond(Matrix _A)`

#### Parameters 

* Matrix _A : input Matrix A with size of (nx1), here, A represents eigenvalues of original Matrix
* Returns condition number for eigenvalues in scalar forms.

#### Example Codes

```c
/* Main Function */
Matrix matA = txt2Mat(path,"prob1_matA");
Matrix lam = createMat(A.rows,1);
lam = lamda(matA);
double con;
con = cond(lam);
/* Main Function Ends */

```

## Curve Fitting, Interpolation

### linearFit()

----



This function gives least square approximation for the linear function. That is, it gives coefficient for the linear least square fitting function f(x)=a0+a1x.

`Matrix linearFit(Matrix _x, Matrix _y)`

#### Parameters

* Matrix _x : Input x data sets, should be a vector form and (nx1) size
* Matrix _y : Input y data sets, should be a vector form and (nx1) size with size of same as _x
* Returns fitting coefficient a0 and a1 in (2x1) matrix form.

#### Example Codes

```c
int main(int argc, char* argv[])
{
	int M = 6;
	double T_array[] = { 30, 40, 50, 60, 70, 80 };
	double P_array[] = { 1.05, 1.07, 1.09, 1.14, 1.17, 1.21 };

	Matrix T = arr2Mat(T_array, M, 1);
	Matrix P = arr2Mat(P_array, M, 1);

	Matrix z = linearFit(T, P);

	printMat(T, "T");
	printMat(P, "P");
	printMat(z, "z");

	system("pause");
	return 0;
}
```

### Residuals()

---

This function serves as residual of linear curve fitting, which refers to 
$$
r_k = y_k - (a_1*x_k+a_0)
$$
By using this function, total error can be found.

`void residual(Matrix _x, Matrix _y,Matrix _r,double n, Matrix _a)`

#### Parameters

* Matrix _x : input datasets which has (nx1) size
* Matrix _y : input datasets which has same size with _x
* Matrix _r : storing residual values in matrix, (nx1) size
* double n : The number of datasets
* Matrix _a : curve fitting coefficients, (2x1) size

At the end of the function, _r matrix will store the residual values

#### Example Codes

```c
int main(int argc, char* argv[])
{
	int M = 6;
	double T_array[] = { 30, 40, 50, 60, 70, 80 };
	double P_array[] = { 1.05, 1.07, 1.09, 1.14, 1.17, 1.21 };

	Matrix T = arr2Mat(T_array, M, 1);
	Matrix P = arr2Mat(P_array, M, 1);
	Matrix z = linearFit(T, P);
	Matrix r = createMat(M,1);
	residual(T,P,r,M,z);

	system("pause");
	return 0;
}
```



### TotalError()

----

This function serves as finding the total error of linear curve fitting by squaring residual of each term

`void TotalError(Matrix _x, Matrix _y, double n, Matrix _a, double Er, Matrix _residual)`

#### Parameters

* Matrix _x : input datasets which has (nx1) size
* Matrix _y : input datasets which has same size with _x
* Matrix _a : curve fitting coefficients, (2x1) size
* double Er : storing total values
* Matrix _residual : residual values, (nx1) size

### curve()

----

This function serves as curve fitting for higher order, such as 2, 3, 4.. etc. 

`Matrix curve(Matrix _x, Matrix _y, int od)`

#### Parameters

* Matrix _x : input datasets which has (nx1) size
* Matrix _y : input datasets which has same size with _x
* int od : order of curve fitting

#### Example Codes

```c
int main(int argc, char* argv[])
{
	int M = 6;
	double T_array[] = { 30, 40, 50, 60, 70, 80 };
	double P_array[] = { 1.05, 1.07, 1.09, 1.14, 1.17, 1.21 };

	Matrix T = arr2Mat(T_array, M, 1);
	Matrix P = arr2Mat(P_array, M, 1);
    int od = 2;
    Matrix c = createMat(M,1);
    c = curve(T,P,od);
    
    return 0;
}

```



### arr2Mat()

---



This function gets array as a input and transform its components as a form of matrix.

`Matrix	arr2Mat(double* _1Darray, int _rows, int _cols)`

#### Parameters

* double* _1Darray : one-dimensional array that has 1 column with n rows. That is,(nx1) size
* int _rows: number of rows of input that user gives
* int _cols : number of columns of input that user gives (mostly 1 columns)

#### Example Codes

```c
	int M = 6;
	double T_array[] = { 30, 40, 50, 60, 70, 80 };
	double P_array[] = { 1.05, 1.07, 1.09, 1.14, 1.17, 1.21 };

	Matrix T = arr2Mat(T_array, M, 1);
	Matrix P = arr2Mat(P_array, M, 1);
```

### linearInterp()

This function gives linear interpolated values for given data. 

`Matrix   linearInterp(Matrix _x, Matrix _y, Matrix _xq)`

#### Parameters

* Matrix _x : input x datasets with length of rows of x, n. it should be (nx1) size
* Matrix _y : input y datasets with length of rows of y, n. it should be (nx1) size
* Matrix _xq : input query input datasets with length of m. it should be (mx1) size
* Returns interpolated value with size of (mx1) vector 

#### Example Codes

```C
	Matrix T = txt2Mat(path, "prob1_vecT");
	Matrix P = txt2Mat(path, "prob1_vecP");
	int n = 21;
	double xq_array[] = { 0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100 };
	Matrix xq = arr2Mat(xq_array, n,1);
	Matrix ypoly = linearInterp(T, P, xq);
```

----

# Numerical Differentiation

## gradient()

Solves for numerical gradient  (dy/dt) from  a set of discrete data

```c
Matrix	gradient(Matrix _x, Matrix _y) 
```

#### **Parameters**

- **x**:  vector **t** in structure Matrix form.  Should be (nx1) vector
- **y**:  vector  **y** in structure Matrix form.  Should be (nx1) vector and same length as t
- Returns **dydt** in structure Matrix form. Output is also (nx1) vector



#### Example code

```c
double t_array[] = { 0,0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0,2.2,2.4,2.6,2.8,3.0,3.2,3.4,3.6,3.8,4.0 };
	double x_array[] = { -5.87, -4.23, -2.55, -0.89, 0.67, 2.09, 3.31, 4.31, 5.06, 5.55, 5.78, 5.77, 5.52, 5.08, 4.46, 3.72, 2.88, 2.00, 1.10, 0.23, -0.59 };                                  
	int len = sizeof(t_array) / sizeof(double);
	Matrix xa = arr2Mat(x_array, len, 1);
	Matrix ta = arr2Mat(t_array, len, 1);
	int n = len;
	Matrix grad=gradient(ta, xa);        
	printMat(grad, "vel");
	Matrix acc = gradient(ta, grad);
```

***



### gradient1D()

---



In this function, instead of using Matrix form, array form of data is used to find gradient of given datasets. 

`void gradient1D(double x[], double y[], double dydx[], int n)`

#### Parameters

* double x[] : Input x datasets as a form of array
* double y[] : Input y datasets as a form of array, length of y should be equal to that of x
* double dydx[] : Output datasets which is calculated and filled in the function
* int n : Length of given datasets
* Returns dydx as a form of 1D array

#### Example Codes

```c
	double t_array[] = { 0,0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0,2.2,2.4,2.6,2.8,3.0,3.2,3.4,3.6,3.8,4.0 };
	double x_array[] = { -5.87, -4.23, -2.55, -0.89, 0.67, 2.09, 3.31, 4.31, 5.06, 5.55, 5.78, 5.77, 5.52, 5.08, 4.46, 3.72, 2.88, 2.00, 1.10, 0.23, -0.59 };                                  
	int len = sizeof(t_array) / sizeof(double);
	double _dydx[21] = {0};
	gradient1D(t_array, x_array, _dydx, len);
```

### func_call()

---



This function serves as function value call corresponding to given input xin

`void func_call(double func(const double x), double xin)`

#### Parameters 

* double func(const double x) : reference function declared in main 
* double xin : reference x value determined in main
* it returns function value as a form of scalar

#### Example Codes

````C
void func_call(double func(const double x), double xin)
{
	double yout = func(xin);
	printf("Y_out by my_func1 = %f\n", yout);

 }
````

### gradientFunc()

----



This function gives gradient of given function, not datasets. thus, input will be  reference function, and xin values.

`Matrix	gradientFunc(double func(const double x), Matrix xin)`

#### Parameters

* double func(const double x) : reference function declared in main
* Matrix xin : reference xin values which has (nx1) length.

#### Example Codes

```c
	double x_inputarr[]= { 		0,0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0,2.2,2.4,2.6,2.8,3.0,3.2,3.4,3.6,3.8,4.0 };
	int len2 = sizeof(x_inputarr) / sizeof(double);
	Matrix xin = arr2Mat(x_inputarr, len, 1);
	Matrix gradientF = gradientFunc(myFuncTest, xin);
```



### newtonRaphsonFunc()

----

In this function, another usage of Newton-Raphson method is implemented : depending on function thus, input will be referenced function, derivative of it, initial x, and tolerance

`double newtonRaphsonFunc(double myfunc(const double x), double mydfunc(const double x), double x0, double tol)`

#### Parameters 

* double myfunc(const double x) : referenced function declared in main
* double mydfunc(const double x) : referenced function's derivative
* double x0 : initial guess of approximation of true solution
* double tol : tolerance that stops iteration when it is within given error interval

#### Example Codes

```c
	double newtonresult;
	double x0 = 2.0;
	double tol = 0.00001;
	double testresults= newtonRaphsonFunc(myFuncTest, mydFuncTest, x0, tol);
	newtonresult=newtonRaphsonFunc(myFunc,mydFunc,x0,tol);
```

---

## Numerical Integration

---

### trapz()

---

This function serves as numerical integration by using trapezoidal method. 

`double trapz(double x[], double y[], int m)`

#### Parameters

* double x[] : input x datasets with length of m
* double y[] : input y datasets with length of m, should have same length with x[]
* int m : length of datasets
* Returns integrated value.

#### Example Codes

```c

	double x[] = { 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60 };
	double y[] = { 0, 3, 8, 20, 33, 42, 40, 48, 60, 12, 8, 4, 3 };
	int M = sizeof(x) / sizeof(x[0]);
	I_trapz = trapz(x, y, M);
```

### IntegrateRect()

---



This function serves as numerical integration by using rectangular method. It has the biggest error from true solution compared to other methods.

`double IntegrateRect(double _x[], double _y[], int _m)`

#### Parameters

* double x[] : input x datasets with length of m
* double y[] : input y datasets with length of m, should have same length with x[]
* int m : length of datasets
* Returns integrated value.

#### Example Codes

```c

	double x[] = { 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60 };
	double y[] = { 0, 3, 8, 20, 33, 42, 40, 48, 60, 12, 8, 4, 3 };
	int M = sizeof(x) / sizeof(x[0]);

	double I_rect = IntegrateRect(x, y, M);
```

### Integral()

---



This function serves as numerical integration using Simpson 13 method.

`double integral(double func(const double x), double a, double b, int n)`

#### Parameters

* double func(const double x) : Reference function declared in main
* double a : lower limit of integral
* double b : upper limit of integral
* int n : number of point between interval : (b-a)/h, where h is difference.
* Returns integrated value.

#### Example Codes

```c
double I_simpson13 = 0;
	int m = 12;
	int a = -1;
	int b = 1;

	// ADD YOUR CODE HERE.   I_simpson13=integral()
	I_simpson13 = integral(myFunc, a, b, m);
```



### Integral38()

---



This function serves as numerical integration using Simpson 38 method.

`double integral38(double func(const double x), double a, double b, int n)`

#### Parameters

* double func(const double x) : Reference function declared in main
* double a : lower limit of integral
* double b : upper limit of integral
* int n : number of point between interval : (b-a)/h, where h is difference.
* Returns integrated value of Simpson 38

#### Example Codes

```c
	int m = 12;
	int a = -1;
	int b = 1;
	double I_simpson38 = 0.0;
	I_simpson38 = integral38(myFunc, a, b, m);
```

### integralMid()

---



This function serves as numerical integration using midpoint integration method.

`double integralMid(double x[], double y[], int m)`

#### Parameters

* double x[] : input x datasets with length of m
* double y[] : input y datasets with length of m, should have same length with x[]
* int m : length of datasets
* Returns integrated value of midpoint integration

#### Example Codes

````c
	double x[] = { 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60 };
	double y[] = { 0, 3, 8, 20, 33, 42, 40, 48, 60, 12, 8, 4, 3 };
	int M = sizeof(x) / sizeof(x[0]);
	double I_middle = integralMid(x, y, M);
````

---------

## First Order Ordinary Differential Equations With Initial Value

---

### odeEU()

---



This function gives the solution for ordinary differential equation with initial value by using Euler's method. 

`void odeEU(double func(const double t), double y[], double t0, double tf, double h)`

#### Parameters

* double func(const double t): reference function declared in main
* double y[] : array that stores result values, with length of (b-a)/h
* double t0 : initial time
* double tf : final time
* double h : arbitrary difference
* Returns result y[i], and needs to plot it in MATLAB

#### Example Codes

```c
	double a =0.0;
	double b =0.1;
	double h = 0.001;
	double y[100] = {};
	odeEU(myfunc, y, a, b, h);
```

### odeEM()

---



This function gives the solution for ordinary differential equation with initial value by using Euler's modified method.

`void odeEM(double func(const double t), double y[], double t0, double tf, double h)`

#### Parameters

* double func(const double t): reference function declared in main
* double y[] : array that stores result values, with length of (b-a)/h
* double t0 : initial time
* double tf : final time
* double h : arbitrary difference
* Returns result y[i], and needs to plot it in MATLAB

#### Example Codes



```c
	double a =0.0;
	double b =0.1;
	double h = 0.001;
	double y1[100] = {};
	odeEM(myfunc, y1, a, b, h);
```

### odeRK2()

---

This function gives This function gives the solution for ordinary differential equation with initial value by using Runge-Kutta method. (Should be used only for 1st order Ordinary Differential Equation)

`void odeRK2(double odeFunc(const double t, const double y), double y[], double t0, double tf, double h, double y0)`

#### Parameters

* double func(const double t, const double y): reference function declared in main
* double y[] : array that stores result values, with length of (b-a)/h
* double t0 : initial time
* double tf : final time
* double h : arbitrary difference
* double y0 : initial value
* Returns result y[i], and needs to plot it in MATLAB

#### Example Codes

```c
	double a = 0;
	double b = 0.1;
	double h = 0.001;
	double y_RK2[200] = { 0.0 };
	double v0 = 0;
	odeRK2(odeFunc_rc, y_RK2, a, b, h, v0);
```

### odeRK4()

----

This function gives This function gives the solution for ordinary differential equation with initial value by using Runge-Kutta method. (Should be used only for 1st order Ordinary Differential Equation)

`void odeRK4(double odeFunc(const double t, const double y), double y[], double t0, double tf, double h, double y0)`

#### Parameters 

* double func(const double t, const double y): reference function declared in main
* double y[] : array that stores result values, with length of (b-a)/h
* double t0 : initial time
* double tf : final time
* double h : arbitrary difference
* double y0 : initial value
* Returns result y[i], and needs to plot it in MATLAB

#### Example Codes

```C
	double a = 0;
	double b = 0.1;
	double h = 0.001;
	double y_RK4[200] = { 0.0 };
	double v0 = 0;
	odeRK4(odeFunc_rc, y_RK2, a, b, h, v0);
```

---

## Second Order Ordinary Differential Equations With Initial Value

---

### sys2RK2()

---

This function gives the solution for ordinary differential equation with initial value by using Runge-Kutta method. (Used for 2nd or higher order Ordinary Differential Equation)

`void sys2RK2(void odeFunc_sys2(const double t, const double Y[], double dYdt[]), double y1[], double y2[], double t0, double tf, double h, double y1_init, double y2_init)`

#### Parameters

* void odeFunc_sys2(const double t, const double Y[], double dYdt[]) : Referenced function declared in main
* double y1[] : y(t) values
* double y2[] : dy/dt values = z(t)
* double t0 : initial condition(left end of interval)
* double tf : Final condition(right end of interval)
* double h : difference
* double y1_init : initial condition of y(t)
* double y2_init : initial condition of y'(t) = z(t)

#### Example Codes

```c
	double t0 = 0;
	double tf = 1;
	h = 0.01;
	N = (tf - t0) / h + 1;
	double y[200] = { 0 };
	double v[200] = { 0 };

	// Initial values
	double y0 = 0;
	v0 = 0.2;

	// ODE solver: RK2

	sys2rk2(odeFunc_mck, y, v, t0, tf, h, y0, v0);
```

### sysRK4()

---

This function gives the solution for ordinary differential equation with initial value by using Runge-Kutta 4 method.

`void sys2RK4(void odeFunc_sys2(const double t, const double Y[], double dYdt[]), double y1[], double y2[], double t0, double tf, double h, double y1_init, double y2_init)`

#### Parameters

* void odeFunc_sys2(const double t, const double Y[], double dYdt[]) : Referenced function declared in main
* double y1[] : y(t) values
* double y2[] : dy/dt values = z(t)
* double t0 : initial condition(left end of interval)
* double tf : Final condition(right end of interval)
* double h : difference
* double y1_init : initial condition of y(t)
* double y2_init : initial condition of y'(t) = z(t)



#### Example Codes

```c
	double t0 = 0;
	double tf = 1;
	h = 0.01;
	N = (tf - t0) / h + 1;
	double y[200] = { 0 };
	double v[200] = { 0 };

	// Initial values
	double y0 = 0;
	v0 = 0.2;

	// ODE solver: RK4

	sys2RK4(odeFunc_mck, y, v, t0, tf, h, y0, v0);
```

