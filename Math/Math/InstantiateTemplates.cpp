//
// <copyright file="InstantiateTemplates.cpp" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
#include <string>
#include <assert.h>
#include "CPUMatrix.cpp"
#include "Matrix.cpp"

#include "..\..\common\include\fileutil.cpp"
#include "..\..\common\include\File.cpp"

    //don't treat it as sample code. some code does not make sense 
    //only used to force compiler to build the code
namespace Microsoft { namespace MSR { namespace CNTK {
#pragma region instantiate all classes (so clients can link to them)

    template <class T>
    void CallEverythingInMatrix()
    {
	    const size_t numRows = 1;
	    const size_t numCols = 1;
	    const bool srcIsColMajor = true;
	    Matrix<T> matx;
        matx.GetCurrentMatrixLocation();        
        matx.TransferFromDeviceToDevice(matx.GetDeviceId(),0);
	    Matrix<T> mat(numRows, numCols);
        const Matrix<T> matt(numRows, numCols);
        T y = matt(0,0);
	    Matrix<T> mat2(numRows, numCols, nullptr, srcIsColMajor);
	    Matrix<T> mat3(mat);  //copy constructor, deep copy
	    Matrix<T> mat4 = mat2;  //assignment operator, deep copy
	    Matrix<T> mat5(mat3); 
        mat5.ColumnElementMultiplyWith(mat3);
        FILE * f=0;
	    Matrix<T> mat6(f, "test");
        std::wstring s;
	    File ff(s,fileOptionsText | fileOptionsReadWrite);
        Matrix<T> MM;
        ff>>MM;
        ff<<MM;
        std::vector<Matrix<T>> v;
        v.insert(v.begin() + 1, Matrix<T>(1,1,-1)); //enable move assignment

        size_t rows = mat.GetNumRows();
        size_t cols = mat.GetNumCols();
        size_t elems = mat.GetNumElements();
        mat.Reshape(numRows, numCols);
        mat.Resize(numRows, numCols);
        mat.IsEmpty();
	    size_t row = 0, col = 0;
        T val = (const T&)mat(row, col);
        mat(row, col) = 0;
        mat.SetValue(val);
        mat.SetValue(mat2);
        mat.SetValue(numRows, numCols, NULL, srcIsColMajor);

        mat.SetDiagonalValue(1);
        mat.SetDiagonalValue(mat2);
        mat.SetUniformRandomValue(0, 1);
        mat.SetGaussianRandomValue(0,1);

        mat2 = mat.Transpose();
        mat2.AssignTransposeOf(mat);
        //mat.InplaceTranspose();

	    T alpha = (T)2;
	    mat += alpha;
        Matrix<T>& newmat2 = mat + alpha; //enable operator+ and move constructor
	    mat += mat2;
        mat3 = mat + mat2;
        mat3.AssignSumOf(alpha, mat);
        mat3.AssignSumOf(mat, mat);
	    mat -= alpha;
        mat3 = mat - alpha;
        mat3.AssignDifferenceOf(alpha, mat);
        mat3.AssignDifferenceOf(mat, alpha);
	    mat -= mat2;
        mat3 = mat - mat2;
        mat3.AssignDifferenceOf(mat, mat2);
	    mat *= alpha;
        mat3 = mat * alpha;
        mat3.AssignProductOf(alpha, mat);
        mat3 = mat * mat2;
        mat = mat3.AssignProductOf(mat, true, mat2, false);
	    mat /= alpha;
        mat3 = mat / alpha;
	    mat ^= alpha;
        mat3 = mat ^ alpha;
        mat3.AssignElementPowerOf(mat, alpha);
        mat3 = mat2.ElementMultiplyWith (mat);
        mat3 = mat2.AssignElementProductOf (mat, mat4);
        mat3.AddElementProductOf(mat,mat4);

        mat3 = mat2.ElementInverse();
        mat3.AssignElementInverseOf (mat2);

        mat3 = mat2.InplaceSigmoid ();
        mat3.AssignSigmoidOf (mat);

        mat3 = mat2.InplaceTanh ();
        mat3.AssignTanhOf (mat2);

	    const bool isColWise = true;
        mat3 = mat2.InplaceSoftmax (isColWise);
        mat3.AssignSoftmaxOf (mat2, isColWise);

        mat3 = mat2.InplaceSqrt ();
        mat3.AssignSqrtOf (mat2);

        mat3 = mat2.InplaceExp ();
        mat3.AssignExpOf (mat2);

        mat3 = mat2.InplaceLog ();
        mat3.AssignLogOf (mat2);

        mat3 = mat2.InplaceAbs ();
        mat3.AssignAbsOf (mat2);

        mat3 = mat2.InplaceTruncateBottom ((T)1);
        mat3 = mat2.InplaceTruncateTop ((T)1);
        mat3 = mat2.SetToZeroIfAbsLessThan ((T)1);

        T sum = mat3.Sum (); //sum of all elements

        mat.IsEqualTo(mat3);
        mat.IsEqualTo(mat3, 0.1f);

        mat.VectorNorm1(mat2, isColWise);
        mat2.AssignVectorNorm1Of(mat2, isColWise);

        mat.VectorNorm2(mat2, isColWise);
        mat2.AssignVectorNorm2Of(mat2, isColWise);

        mat.VectorNormInf(mat2, isColWise);
        mat2.AssignVectorNormInfOf(mat2, isColWise);

        T frob = mat.FrobeniusNorm();
        T norm = mat.MatrixNormInf();
        mat.VectorMax(mat2, mat3, isColWise);
        mat.VectorMin(mat2, mat3, isColWise);

        mat3.AssignInnerProductOf(mat, mat2, isColWise);
        mat3.AddWithScaleOf(alpha, mat);

        mat.Print("test", 0,1,0,1);
        mat.Print();
        mat.Print("test");

        mat.ReadFromFile(f, "test");
        mat.WriteToFile(f, "test");

	    T beta = (T)1.0;
	    const bool transposeA = true;
	    const bool transposeB = false;
        Matrix<T>::MultiplyAndWeightedAdd(alpha, mat2, transposeA, mat3, transposeB, beta, mat4);
        Matrix<T>::MultiplyAndAdd(mat2, transposeA, mat3, transposeB, mat4);
        Matrix<T>::Multiply(mat2, transposeA, mat3, transposeB, mat4);
        Matrix<T>::Multiply(mat2, mat3, mat4);
        Matrix<T>::ScaleAndAdd(alpha, mat2, mat3);
        Matrix<T>::Scale(alpha, mat2);
        Matrix<T>::Scale(alpha, mat2, mat);
        Matrix<T>::InnerProduct(mat2, mat3, mat4, isColWise);
        Matrix<T>::ElementWisePower (0, mat, mat3);
        Matrix<T>::AreEqual(mat, mat3);
        Matrix<T>::AreEqual(mat, mat3, (T)0.1);

        Matrix<T>::Ones(2,3);
        Matrix<T>::Zeros(2,3);
        Matrix<T>::Eye(2);
        Matrix<T>::RandomUniform(2, 3,0, 1);
        Matrix<T>::RandomGaussian(2, 3,0, 1);
    }


    template <class T>
    void CallEverythingInCPUMatrix()
    {
	    const size_t numRows = 1;
	    const size_t numCols = 1;
	    const bool srcIsColMajor = true;
	    CPUMatrix<T> matx;
	    CPUMatrix<T> mat(numRows, numCols);
	    CPUMatrix<T> mat2(numRows, numCols, nullptr, srcIsColMajor);
	    CPUMatrix<T> mat3(mat);  //copy constructor, deep copy
	    CPUMatrix<T> mat4 = mat2;  //assignment operator, deep copy
	    CPUMatrix<T> mat5(mat3); 
        mat5.ColumnElementMultiplyWith(mat3);
        FILE * f=0;
        std::wstring s;
	    File ff(s,fileOptionsText | fileOptionsReadWrite);
        CPUMatrix<T> MM;
        ff>>MM;
        ff<<MM;
        
        CPUMatrix<T> mat6(f, "test");
        std::vector<CPUMatrix<T>> v;
        v.insert(v.begin() + 1, CPUMatrix<T>(1,1)); //enable move assignment

        size_t rows = mat.GetNumRows();
        size_t cols = mat.GetNumCols();
        size_t elems = mat.GetNumElements();
        mat.Reshape(numRows, numCols);
        mat.Resize(numRows, numCols);
        mat.IsEmpty();
	    size_t row = 0, col = 0;
	    T val = mat(row, col);
        mat(row, col) = 0;
        mat.SetValue(val);
        mat.SetValue(mat2);
        mat.SetValue(numRows, numCols, NULL, srcIsColMajor);

        mat.SetDiagonalValue(1);
        mat.SetDiagonalValue(mat2);
        mat.SetUniformRandomValue(0, 1);
        mat.SetGaussianRandomValue(0,1);

        mat2 = mat.Transpose();
        mat2.AssignTransposeOf(mat);
        //mat.InplaceTranspose();

	    T alpha = (T)2;
	    mat += alpha;
        CPUMatrix<T>& newmat2 = mat + alpha; //enable operator+ and move constructor
	    mat += mat2;
        mat3 = mat + mat2;
        mat3.AssignSumOf(alpha, mat);
        mat3.AssignSumOf(mat2, mat);
	    mat -= alpha;
        mat3 = mat - alpha;
        mat3.AssignDifferenceOf(alpha, mat);
        mat3.AssignDifferenceOf(mat, alpha);
	    mat -= mat2;
        mat3 = mat - mat2;
        mat3.AssignDifferenceOf(mat, mat2);
	    mat *= alpha;
        mat3 = mat * alpha;
        mat3.AssignProductOf(alpha, mat);
        mat3 = mat * mat2;
        mat = mat3.AssignProductOf(mat, true, mat2, false);
	    mat /= alpha;
        mat3 = mat / alpha;
	    mat ^= alpha;
        mat3 = mat ^ alpha;
        mat3.AssignElementPowerOf(mat, alpha);
        mat3 = mat2.ElementMultiplyWith (mat);
        mat3 = mat2.AssignElementProductOf (mat, mat4);
        mat3 = mat2.AddElementProductOf (mat, mat4);
        mat2.ColumnElementMultiplyWith(mat);

        mat3 = mat2.ElementInverse();
        mat3.AssignElementInverseOf (mat2);

        mat3 = mat2.InplaceSigmoid ();
        mat3.AssignSigmoidOf (mat);

        mat3 = mat2.InplaceTanh ();
        mat3.AssignTanhOf (mat2);

	    const bool isColWise = true;
        mat3 = mat2.InplaceSoftmax (isColWise);
        mat3.AssignSoftmaxOf (mat2, isColWise);

        mat3 = mat2.InplaceSqrt ();
        mat3.AssignSqrtOf (mat2);

        mat3 = mat2.InplaceExp ();
        mat3.AssignExpOf (mat2);

        mat3 = mat2.InplaceLog ();
        mat3.AssignLogOf (mat2);

        mat3 = mat2.InplaceAbs ();
        mat3.AssignAbsOf (mat2);

        mat3 = mat2.InplaceTruncateBottom ((T)1);
        mat3 = mat2.InplaceTruncateTop ((T)1);
        mat3 = mat2.SetToZeroIfAbsLessThan ((T)1);

        T sum = mat3.Sum (); //sum of all elements

        mat.IsEqualTo(mat3);
        mat.IsEqualTo(mat3, 0.1f);

        mat.VectorNorm1(mat2, isColWise);
        mat2.AssignVectorNorm1Of(mat2, isColWise);

        mat.VectorNorm2(mat2, isColWise);
        mat2.AssignVectorNorm2Of(mat2, isColWise);

        mat.VectorNormInf(mat2, isColWise);
        mat2.AssignVectorNormInfOf(mat2, isColWise);

        T frob = mat.FrobeniusNorm();
        T norm = mat.MatrixNormInf();
        norm = mat.MatrixNorm1();
        mat.AssignSignOf(mat2);
        mat.AddSignOf(mat2);

        mat.VectorMax(mat2, mat3, isColWise);
        mat.VectorMin(mat2, mat3, isColWise);

        mat3.AssignInnerProductOf(mat, mat2, isColWise);
        mat3.AddWithScaleOf(alpha, mat);

        mat.Print("test", 0,1,0,1);
        mat.Print();
        mat.Print("test");

        mat.ReadFromFile(f, "test");
        mat.WriteToFile(f, "test");

	    T beta = (T)1.0;
	    const bool transposeA = true;
	    const bool transposeB = false;
        CPUMatrix<T>::MultiplyAndWeightedAdd(alpha, mat2, transposeA, mat3, transposeB, beta, mat4);
        CPUMatrix<T>::MultiplyAndAdd(mat2, transposeA, mat3, transposeB, mat4);
        CPUMatrix<T>::Multiply(mat2, transposeA, mat3, transposeB, mat4);
        CPUMatrix<T>::Multiply(mat2, mat3, mat4);
        CPUMatrix<T>::ScaleAndAdd(alpha, mat2, mat3);
        CPUMatrix<T>::Scale(alpha, mat2);
        CPUMatrix<T>::Scale(alpha, mat2, mat);
        CPUMatrix<T>::InnerProduct(mat2, mat3, mat4, isColWise);
        CPUMatrix<T>::InnerProductOfMatrices(mat,mat2);
        CPUMatrix<T>::ElementWisePower (0, mat, mat3);
        CPUMatrix<T>::AreEqual(mat, mat3);
        CPUMatrix<T>::AreEqual(mat, mat3, (T)0.1);

        CPUMatrix<T>::Ones(2,3);
        CPUMatrix<T>::Zeros(2,3);
        CPUMatrix<T>::Eye(2);
        CPUMatrix<T>::RandomUniform(2, 3,0, 1);
        CPUMatrix<T>::RandomGaussian(2, 3,0, 1);
    }

    void InstantiateAllCPUMatrixMethods()
    {
	    CallEverythingInCPUMatrix<float>();
	    CallEverythingInCPUMatrix<double>();

        CallEverythingInMatrix<float>();
	    CallEverythingInMatrix<double>();
    }
#pragma endregion instantiate all classes 
}}}