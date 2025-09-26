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

// Original file: Microsoft.Research\RegressionTest\ClousotTests\Sources\TestContainers\BuiltInFunctions.cs
// Modified for Index Checker evaluation: removed parts that are not representable or are irrelevant to array bounds

#define CONTRACTS_FULL

using System;
using System.Linq;
using System.Diagnostics.Contracts;
using Microsoft.Research.ClousotRegression;

namespace Linq
{
  public class HandleCount
  {
	[ClousotRegressionTest]
	[RegressionOutcome(Outcome=ProofOutcome.Top,Message="Array access might be above the upper bound. The static checker determined that the condition '1 < a.Length' should hold on entry. Nevertheless, the condition may be too strong for the callers. If you think it is ok, add a precondition to document it: Contract.Requires(1 < a.Length);",PrimaryILOffset=16,MethodILOffset=0)]
	static public  int FooWrong(int[] a)
    {
		// Here we use the fact that a.Length == a.Count
      if (a == null || a.Count() == 2)
      {
        return -1;
      }

      return a[1];
    }
  }
}
