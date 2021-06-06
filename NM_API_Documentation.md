`#include "myNM.h"`

## Nonlinear Solver 



### Bisection()

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



# Linear Solver

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

#### Example Codes

```c
Matrix matA = txt2Mat(path, "prob1_matA");
Matrix inv = inv(matA);
```



### QRFact()

---

This function serves as QR factorization of given Matrix A. This function implements house hold matrix to find QR factorization. This is commonly used to find eigenvalues and eigenvectors.

`void QRFact(Matrix _A)`

#### Parameters

* Matrix _A : input Matrix A with size of (nxn)

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

####  Example Codes 

```c
Matrix matA = txt2Mat(path, "prob1_matA");
lam = lamda(matA);
```



## gaussElim()

solves for vector **x** from  Ax=b,  a linear system problem  

```c
void	gaussElim(Matrix _A, Matrix _B, Matrix* _U, Matrix* _B_out);
```

#### **Parameters**

- **A**:  Matrix **A** in structure Matrix form.  Should be (nxn) square.

- **B**:  vector  **b** in structure Matrix form.  Should be (nx1) 

  

## solveLinear()

solves for vector **x** from  Ax=b,  a linear system problem  

```c
extern Matrix solveLinear(Matrix _A, Matrix _b, char* _method)
```

#### **Parameters**

- **A**:  Matrix **A** in structure Matrix form.  Should be (nxn) square.

- **b**:  vector  **b** in structure Matrix form.  Should be (nx1) 

- **method:  character type,** 

  - **'lu' :** LU decomposition
  - **'gauss':** Gauss elimination

  

#### Example code

```C
double A_array[] = { 1, 3, -2, 4,		2, -3, 3, -1,		-1, 7, -4, 2,		-1, 7, -4, 2 };
double b_array[] = { -11,		6,		-9,		15 };

Matrix matA = arr2Mat(A_array, M, N);
Matrix vecb = arr2Mat(b_array, M, 1);

Matrix x_lu = solveLinear(matA, vecb, "LU");
Matrix invA_gj = inv(matA, "gj");
Matrix invA_lu = inv(matA, "LU");
```





***



# Numerical Differentiation

## gradient()

Solves for numerical gradient  (dy/dt) from  a set of discrete data

```c
Matrix	gradient(Matrix _t, Matrix _y);
```

#### **Parameters**

- **t**:  vector **t** in structure Matrix form.  Should be (nx1) vector
- **y**:  vector  **y** in structure Matrix form.  Should be (nx1) vector and same length as t
- Returns **dydt** in structure Matrix form. Output is also (nx1) vector



#### Example code

```c
Matrix t = txt2Mat("", "Q1_vect");
Matrix x = txt2Mat("", "Q1_vecx");

Matrix vel = gradient(t, x);
Matrix acc = gradient(t, vel);

printMat(t, "t");
printMat(x, "x");
printMat(vel, "vel");
printMat(acc, "acc");
```

See full example code:  [TutorialDifferentiation.cpp](https://github.com/ykkimhgu/tutorial-NM/blob/main/samples/Tutorial-Differentiation.cpp)

