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

// Original file: Microsoft.Research\RegressionTest\ClousotTests\Sources\Enum\EnumTest.cs
// Modified for Index Checker evaluation: removed parts that are not representable or are irrelevant to array bounds

#define CONTRACTS_FULL

using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using Microsoft.Research.ClousotRegression;


namespace IsDefinedAnalysis
{
  public enum Bikes { DeRosa, Specialized, Trek, Daccordi, Colnago }

  public class MyClass
  {
	
    [ClousotRegressionTest]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"Lower bound access ok",PrimaryILOffset=29,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.Top,Message=@"Array access might be above the upper bound",PrimaryILOffset=29,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"Assignment to an enum value ok",PrimaryILOffset=29,MethodILOffset=0)]
    public static void SetBikeArray(Bikes[] bikes, int x)
    {
      Contract.Requires(Enum.IsDefined(typeof(Bikes), x));

      bikes[0] = (Bikes)x;
    }

    [ClousotRegressionTest]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"Lower bound access ok",PrimaryILOffset=3,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.Top,Message=@"Array access might be above the upper bound",PrimaryILOffset=3,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.Top,Message=@"The assigned value may not be in the range defined for this enum value",PrimaryILOffset=3,MethodILOffset=0)]
    public static void SetBikeArrayNoPre(Bikes[] bikes, int x)
    {
      bikes[0] = (Bikes)x;
    }

  }
}


namespace TestAssumptionsInTheEnumAnalysis
{
   class Test
   {
    public enum SomeEnum { A = -12, B = 0, C = 3, D = 7 }

    [ClousotRegressionTest]
	[RegressionOutcome(Outcome=ProofOutcome.True,Message="Lower bound access ok",PrimaryILOffset=39,MethodILOffset=0)]
	[RegressionOutcome(Outcome=ProofOutcome.True,Message="Upper bound access ok",PrimaryILOffset=39,MethodILOffset=0)]
	[RegressionOutcome(Outcome=ProofOutcome.True,Message="Assignment to an enum value ok",PrimaryILOffset=40,MethodILOffset=0)]
    public int Switch(SomeEnum[] a, int i)
    {
      Contract.Requires(a != null);
      Contract.Requires(i >= 0);
      Contract.Requires(i < a.Length);

      var count = 0;

      //for (var i = 0; i < a.Length; i++)
      {
        switch (a[i])
        {
          case SomeEnum.A:
          case SomeEnum.B:
            count += 1;
            break;

          case SomeEnum.C:
          case SomeEnum.D:
            count += 2;
            break;

          default:
            break;
        }
      }
      return count;
    }
	}

}