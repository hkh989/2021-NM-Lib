## 2021-NM-Lib

### Plane Fitting

---

#### Main

```c
#include "../../../include/myNM.h"

int main(int argc, char* argv[])
{
	/*Problem 1: Plane Fitting Starts*/
	Matrix A = txt2Mat(path, "Q1_matA");
	Matrix B = txt2Mat(path, "Q1_matB");
	Matrix res = PlaneFitting(A, B);
	printMat(res, "Coefficient");
	/* Problem 1: Plane Fitting Ends */
	/* Problem 2 :Aiming and Tilting angle Finder Starts*/
	Matrix plane = createMat(1, res.rows+1);
	plane.at[0][0] = res.at[1][0], plane.at[0][1] = 1, plane.at[0][2] = res.at[0][0],plane.at[0][3] = res.at[2][0];
	Matrix angle = createMat(1, res.rows);
	printMat(plane, "Normal Vectors Parameters");
	angle = AngleTrack(plane);
	printf("Alpha = %f \t Aiming Angle =  %f \t Tilt_Angle = %f\n", angle.at[0][0], angle.at[3][0], angle.at[4][0]);
	/* Problem 2:Aiming and Tilting angle Finder Ends*/
	/* Problem 3:Linear Interpolation to generate 3D datasets Starts*/
	Matrix inter = linearinterpmod(A, B,plane);
	printMat(inter, "zinter");
	return 0; 


}

```

#### Plane Fitting Function

---



```C
Matrix PlaneFitting(Matrix A, Matrix B)
{
	Matrix Atrans = createMat(A.cols, A.rows);
	transpose(A, Atrans);
	Matrix Mult = createMat(Atrans.rows, A.cols);
	multimat(Atrans, A, Mult);
	Matrix inver = createMat(Mult.rows, Mult.cols);
	initMat(inver, 0);
	inverse2(Mult, inver);
	Matrix Mult1 = createMat(inver.rows, Atrans.cols);
	Matrix res = createMat(Mult1.rows, B.cols);
	multimat(inver, Atrans,Mult1);
	multimat(Mult1, B, res);
	

	return res;

}

```

#### Angle Track Function

---

```C
Matrix AngleTrack(Matrix A)
{
	double a = 0, b = 0, c = 0;
	double alpha = 0, beta = 0, gamma = 0;
	Matrix Plane_Norm = createMat(1, A.cols);
	double aiming_angle = 0 , tilt_plane = 0;
	double datasets[5] = { 0.0,};
	for (int i = 0; i < A.cols; i++)
	{
		Plane_Norm.at[0][i] = A.at[0][i] / EuclidNorm(A);
	}
	a = Plane_Norm.at[0][0], b = Plane_Norm.at[0][1], c = Plane_Norm.at[0][2];
	alpha = atan(c / b) * 180 / PI;
	beta = atan(a / c) * 180 / PI; 
	gamma = atan(b / a) * 180 / PI + 180;
	printf("Alpha : %f\t Beta : %f \t Gamma : %f\n", alpha, beta, gamma);
	printf("\n");
	printf("\n");
	aiming_angle = 90 - gamma;// in degree
	tilt_plane = alpha;
	datasets[0] = alpha, datasets[1] = beta, datasets[2] = gamma, datasets[3] = aiming_angle, datasets[4] = tilt_plane;
	Matrix data = arr2Mat(datasets, 5, 1);
	return data;
}

```

#### 3D Interpolation

---

```c
Matrix linearinterpmod(Matrix A, Matrix B,Matrix C)
{
	int diff = 30;// total datasets/division
	int division = 3;//The number of division of matrix
	int count = 0;
	double a = 0, b = 0, c = 0;
	Matrix x = createMat(A.rows, 1);//A(:,1) = Z
	Matrix y = createMat(A.rows, 1);//A(:,2) = X
	Matrix z = createMat(B.rows, 1);//B = Y
	Matrix sX = createMat(x.rows/division, division);
	initMat(sX, 0);
	Matrix sY = createMat(y.rows/division, division);
	initMat(sY, 0);
	Matrix sZ = createMat(z.rows/division, division);
	initMat(sZ, 0);
	Matrix xinter = createMat(x.rows / division, division - 1);
	initMat(xinter, 0);
	Matrix yinter = createMat(y.rows / division, division - 1);
	initMat(yinter, 0);
	Matrix zinter = createMat(z.rows / division, division - 1);
	initMat(zinter, 0);
	for (int i = 0; i < A.rows; i++)
	{
		x.at[i][0] = A.at[i][1];
		y.at[i][0] = B.at[i][0]*-1;
		z.at[i][0] = A.at[i][0];
	}
	

	for (int i = 0; i < division; i++)
	{
		for (int j = 0; j < diff; j++)
		{
			sX.at[j][i] = x.at[j + diff * i][0];//xq
			sY.at[j][i] = y.at[j + diff * i][0];//yq x,y,sX,sY,z
			sZ.at[j][i] = z.at[j + diff * i][0];
		
		}
	}

	for (int i = 0; i < division - 1; i++)
	{
		for (int j = 0; j < sX.rows; j++)
		{
			xinter.at[j][i] = (sX.at[j][i] + sX.at[j][i + 1]) / 2.0;
			yinter.at[j][i] = (sY.at[j][i] + sY.at[j][i + 1]) / 2.0;
		}
	}

	for (int i = 0; i < sX.rows; i++)
	{
		for (int j = 0; j < sX.cols; j++)
		{
			a = sX.at[i][j] - sX.at[i][j + 1], b = sY.at[i][j] - sY.at[i][j + 1];
			c = sZ.at[i][j] - sZ.at[i][j + 1];
			if (a < pow(10, -7))
			{
				zinter.at[i][j] = c * (yinter.at[i][j] - sY.at[i][j]) / b + sZ.at[i][j];

			}
			else if (c < pow(10, -7))
			{
				zinter.at[i][j] = sZ.at[i][j];
			}
			else
			{
				zinter.at[i][j] = c * (xinter.at[i][j] - sX.at[i][j]) / a + sZ.at[i][j];
			}
		}
	}

	return zinter;

}

```

