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

// Original file: Microsoft.Research\RegressionTest\ClousotTests\Sources\TestBooleanConnectives\TestDisjunctions.cs
// Modified for Index Checker evaluation: removed parts that are not representable or are irrelevant to array bounds

using System;

using System.Diagnostics.Contracts;
using Microsoft.Research.ClousotRegression;

namespace BasicTests
{
  public class Disjunctions
  {
    public int[] Arr(int[] a, int len)
    {
      Contract.Requires(a != null);
      Contract.Requires(len >= 0);

      Contract.Ensures(a.Length != len || Contract.Result<int[]>().Length == a.Length);
      Contract.Ensures(a.Length == len || Contract.Result<int[]>().Length == len);

      if (a.Length == len)
        return a;
      else
        return new int[len];
    }

    // ok
    [ClousotRegressionTest]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = @"Array creation : ok", PrimaryILOffset = 2, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = @"valid non-null reference (as receiver)", PrimaryILOffset = 11, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message="valid non-null reference (as array)", PrimaryILOffset=18, MethodILOffset=0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = @"requires is valid", PrimaryILOffset = 7, MethodILOffset = 11)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = @"requires is valid", PrimaryILOffset = 19, MethodILOffset = 11)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = @"assert is valid", PrimaryILOffset = 23, MethodILOffset = 0)]
    public void Use1()
    {
      var a = new int[10];

      var call = Arr(a, 3);
      // Simplify the postconditions to call.Length == 3

      Contract.Assert(call.Length == 3);
    }

    // ok
    [ClousotRegressionTest]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = @"Array creation : ok", PrimaryILOffset = 22, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = @"valid non-null reference (as receiver)", PrimaryILOffset = 31, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.Top, Message = @"Possible use of a null array 'call'", PrimaryILOffset = 38, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = @"requires is valid", PrimaryILOffset = 7, MethodILOffset = 31)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = @"requires is valid", PrimaryILOffset = 19, MethodILOffset = 31)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = @"assert is valid", PrimaryILOffset = 43, MethodILOffset = 0)]
    public void Use2(int len, int newSize)
    {
      Contract.Requires(newSize >= 0);
      Contract.Requires(len > newSize);

      var a = new int[len];

      var call = Arr(a, newSize);
      // Simplify the postconditions to call.Length == newSize

      Contract.Assert(call.Length == newSize);
    }

    // ok
    [ClousotRegressionTest]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = @"Array creation : ok", PrimaryILOffset = 22, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = @"valid non-null reference (as receiver)", PrimaryILOffset = 31, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.Top, Message = @"Possible use of a null array 'call'", PrimaryILOffset = 38, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = @"requires is valid", PrimaryILOffset = 7, MethodILOffset = 31)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = @"requires is valid", PrimaryILOffset = 19, MethodILOffset = 31)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = @"assert is valid", PrimaryILOffset = 43, MethodILOffset = 0)]
    public void Use3(int len, int newSize)
    {
      Contract.Requires(len >= 0);
      Contract.Requires(newSize > len);

      var a = new int[len];

      var call = Arr(a, newSize);
      // Simplify the postconditions to call.Length == newSize

      Contract.Assert(call.Length == newSize);
    }

    // ok
    [ClousotRegressionTest]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = @"Array creation : ok", PrimaryILOffset = 25, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = @"valid non-null reference (as receiver)", PrimaryILOffset = 34, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.Top, Message = @"Possible use of a null array 'call'", PrimaryILOffset = 41, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = @"requires is valid", PrimaryILOffset = 7, MethodILOffset = 34)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = @"requires is valid", PrimaryILOffset = 19, MethodILOffset = 34)]
    [RegressionOutcome(Outcome = ProofOutcome.Top, Message = @"assert unproven", PrimaryILOffset = 46, MethodILOffset = 0)]
    public void Use4(int len, int newSize)
    {
      Contract.Requires(len >= 0);
      Contract.Requires(newSize >= 0);

      var a = new int[len];

      var call = Arr(a, newSize);
      // Cannot simplify the postcondition

      Contract.Assert(call.Length == newSize);
    }
  }
 }

namespace UserRepro
{

  public class ForAllTest
  {
	
	[ClousotRegressionTest]
	[RegressionOutcome(Outcome=ProofOutcome.True,Message="Lower bound access ok",PrimaryILOffset=77,MethodILOffset=0)]
	[RegressionOutcome(Outcome=ProofOutcome.True,Message="Upper bound access ok",PrimaryILOffset=77,MethodILOffset=0)]
	[RegressionOutcome(Outcome=ProofOutcome.True,Message="valid non-null reference (as field receiver)",PrimaryILOffset=8,MethodILOffset=0)]
#if !CLOUSOT2
		[RegressionOutcome(Outcome=ProofOutcome.True,Message="valid non-null reference (as field receiver)",PrimaryILOffset=14,MethodILOffset=0)]
		[RegressionOutcome(Outcome=ProofOutcome.True,Message="valid non-null reference (as field receiver)",PrimaryILOffset=23,MethodILOffset=0)]
#endif
	[RegressionOutcome(Outcome=ProofOutcome.True,Message="valid non-null reference (as array)",PrimaryILOffset=28,MethodILOffset=0)]
	[RegressionOutcome(Outcome=ProofOutcome.True,Message="valid non-null reference (as field receiver)",PrimaryILOffset=58,MethodILOffset=0)]
	[RegressionOutcome(Outcome=ProofOutcome.True,Message="valid non-null reference (as field receiver)",PrimaryILOffset=91,MethodILOffset=0)]
	[RegressionOutcome(Outcome=ProofOutcome.True,Message="valid non-null reference (as array)",PrimaryILOffset=96,MethodILOffset=0)]
	[RegressionOutcome(Outcome=ProofOutcome.True,Message="valid non-null reference (as field receiver)",PrimaryILOffset=71,MethodILOffset=0)]
	[RegressionOutcome(Outcome=ProofOutcome.True,Message="valid non-null reference (as array)",PrimaryILOffset=77,MethodILOffset=0)]
	[RegressionOutcome(Outcome=ProofOutcome.True,Message="valid non-null reference (as receiver)",PrimaryILOffset=78,MethodILOffset=0)]
	public int SumLengths(string[] input)
    {
      Contract.Requires(input == null || Contract.ForAll(0, input.Length, j => input[j] != null));

      var c = 0;
      if (input != null)
      {
        for (var i = 0; i < input.Length; i++)
        {
          c += input[i].Length;
        }
      }

      return c;
    }
  }
}
