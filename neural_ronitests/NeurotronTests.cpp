#include "stdafx.h"
#include "CppUnitTest.h"

#include "..\neural_roniwork\Types.h"
#include "..\neural_roniwork\Neurotron.h"
#include "..\neural_roniwork\Neurotron.cpp"
#include "..\neural_roniwork\NeurotronDataset.h"
#include "..\neural_roniwork\NeurotronDataset.cpp"

#include <Eigen/Dense>

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace neural_ronitests
{		
	TEST_CLASS(NeurotronTests)
	{
	public:
		TEST_METHOD(CountMatchErrAreEqual)
		{
			Eigen::MatrixXd testi1(5, 3);
			Eigen::MatrixXd testi2(5, 1);

			testi1(0, 0) = 0;	testi1(0, 1) = 0;	testi1(0, 2) = 1;
			testi1(1, 0) = 1;	testi1(1, 1) = 0;	testi1(1, 2) = 0;
			testi1(2, 0) = 0;	testi1(2, 1) = 1;	testi1(2, 2) = 0;
			testi1(3, 0) = 1;	testi1(3, 1) = 0;	testi1(3, 2) = 0;
			testi1(4, 0) = 0;	testi1(4, 1) = 0;	testi1(4, 2) = 1;

			testi2(0, 0) = 4;
			testi2(1, 0) = 1;
			testi2(2, 0) = 2;
			testi2(3, 0) = 1;
			testi2(4, 0) = 4;

			Assert::AreEqual(0, nrt::count_match_err(testi1, testi2), L"count_match_err matches", LINE_INFO());

			testi2(0, 0) = 2;
			testi2(3, 0) = 4;

			Assert::AreEqual(2, nrt::count_match_err(testi1, testi2), L"count_match_err matches", LINE_INFO());
		}

	};
}