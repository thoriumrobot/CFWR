// CodeContracts
// 
// Copyright (c) Microsoft Corporation
// 
// All rights reserved. 
// 
// MIT License
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

// Original file: Microsoft.Research\RegressionTest\ClousotTests\Sources\BasicInfrastructure\GenericTests.cs
// Modified for Index Checker evaluation: removed parts that are not representable or are irrelevant to array bounds

using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using Microsoft.Research.ClousotRegression;

// this outcome is due to AnalysisInfrastructure9 compiling all tests in the same mode. Not necessary in dev10 tests.
#if !NETFRAMEWORK_3_5 && !NETFRAMEWORK_4_0
[assembly:RegressionOutcome("Method 'BrianRepro.MyList`1.get_Item(System.Int32)' has custom parameter validation but assembly mode is not set to support this. It will be treated as Requires<E>.")]
#endif

namespace BrianRepro
{

  class CooperativeEvent
  {
    /// <summary>
    /// Disabled for cci2 because it depends on inference of getCount. Somehow cci2 is not providing correct
    /// attributes on getCount, and thus we fail here.
    /// </summary>
    [ClousotRegressionTest]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "Array creation : ok", PrimaryILOffset = 7, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "Array creation : ok", PrimaryILOffset = 18, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "Lower bound access ok", PrimaryILOffset = 32, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "Upper bound access ok", PrimaryILOffset = 32, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as array)", PrimaryILOffset = 48, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as array)", PrimaryILOffset = 32, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as receiver)", PrimaryILOffset = 35, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.Top, Message = @"requires unproven: _size >= 0", PrimaryILOffset = 13, MethodILOffset = 2)]
    [RegressionOutcome(Outcome = ProofOutcome.Top, Message = "Possibly calling a method on a null reference \'waitedOnTaskList\'", PrimaryILOffset = 2, MethodILOffset = 0)]
#if CLOUSOT2
//	#if NETFRAMEWORK_4_0
		[RegressionOutcome(Outcome=ProofOutcome.True,Message="requires is valid",PrimaryILOffset=1,MethodILOffset=35)]
//	#else
//		[RegressionOutcome(Outcome = ProofOutcome.True, Message = "requires is valid", PrimaryILOffset = 8, MethodILOffset = 35)]
//	#endif
#else
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "requires is valid", PrimaryILOffset = 20, MethodILOffset = 35)]
#endif
    public void Foo(MyList<int> waitedOnTaskList)
    {
      CooperativeEvent[] cooperativeEventArray = new CooperativeEvent[waitedOnTaskList.Count];

      int[] arr = new int[MyList<int>.AnInt()];

      for (int index = 0; index < cooperativeEventArray.Length; index++)
      {
        cooperativeEventArray[index] = null;
        int x = waitedOnTaskList[index];
      }
    }


  }

  class MyList<T>
  {
    [ContractPublicPropertyName("Count")]
    private int _size = 0;

    public int Count
    {
      [ClousotRegressionTest("regular")]
      [RegressionOutcome(Outcome = ProofOutcome.True, Message = "ensures is valid", PrimaryILOffset = 30, MethodILOffset = 46)]
      [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as field receiver)", PrimaryILOffset = 2, MethodILOffset = 0)]
      [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as field receiver)", PrimaryILOffset = 37, MethodILOffset = 0)]
      get
      {
        Contract.Requires(_size >= 0);
        Contract.Ensures(Contract.Result<int>() >= 0);
        return _size;
      }
    }

    [ClousotRegressionTest("regular")]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "ensures is valid", PrimaryILOffset = 12, MethodILOffset = 24)]
    static public int AnInt()
    {
      Contract.Ensures(Contract.Result<int>() >= 0);
      return 23;
    }

    public T this[int index]
    {
      [ClousotRegressionTest("regular")]
      [RegressionOutcome(Outcome = ProofOutcome.True, Message = @"valid non-null reference (as field receiver)", PrimaryILOffset = 3, MethodILOffset = 0)]
      get
      {
        // Following trick can reduce the range check by one
        if ((uint)index >= (uint)_size)
        {
          throw new Exception();
        }
        Contract.EndContractBlock();

        // Do nothing ...
        return default(T);
      }

    }

  }
}
