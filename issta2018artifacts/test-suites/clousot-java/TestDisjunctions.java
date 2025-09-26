//CodeContracts
//
//Copyright (c) Microsoft Corporation
//
//All rights reserved. 
//
//MIT License
//
//Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
//
//The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
//
//THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import org.checkerframework.checker.index.qual.NonNegative;
import org.checkerframework.checker.index.qual.SameLen;
import org.checkerframework.common.value.qual.IntVal;

class BasicTests_Disjunctions
	  {
	    public int[] Arr(int[] a, @NonNegative int len)
	    {
	      //N: Contract.Requires(a != null);
	      //Y: Contract.Requires(len >= 0);

	      //N: Contract.Ensures(a.Length != len || Contract.Result<int[]>().Length == a.Length);
	      //N: Contract.Ensures(a.Length == len || Contract.Result<int[]>().Length == len);

	      if (a.length == len)
	        return a;
	      else
	        return new int[len];
	    }

	    // ok
	    /*[ClousotRegressionTest]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = @"Array creation : ok", PrimaryILOffset = 2, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = @"valid non-null reference (as receiver)", PrimaryILOffset = 11, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message="valid non-null reference (as array)", PrimaryILOffset=18, MethodILOffset=0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = @"requires is valid", PrimaryILOffset = 7, MethodILOffset = 11)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = @"requires is valid", PrimaryILOffset = 19, MethodILOffset = 11)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = @"assert is valid", PrimaryILOffset = 23, MethodILOffset = 0)]*/
		    public void Use1()
	    {
		    	int[] a = new int[10];

	      int[] call = Arr(a, 3);
	      // Simplify the postconditions to call.Length == 3

	      //Y: Contract.Assert(call.length == 3);
	      @IntVal(3) int _ = call.length;
	    }

	    // ok  [ClousotRegressionTest]
		    /*[RegressionOutcome(Outcome = ProofOutcome.True, Message = @"Array creation : ok", PrimaryILOffset = 22, MethodILOffset = 0)]
		    	    [RegressionOutcome(Outcome = ProofOutcome.True, Message = @"valid non-null reference (as receiver)", PrimaryILOffset = 31, MethodILOffset = 0)]
		    	    [RegressionOutcome(Outcome = ProofOutcome.Top, Message = @"Possible use of a null array 'call'", PrimaryILOffset = 38, MethodILOffset = 0)]
		    	    [RegressionOutcome(Outcome = ProofOutcome.True, Message = @"requires is valid", PrimaryILOffset = 7, MethodILOffset = 31)]
		    	    [RegressionOutcome(Outcome = ProofOutcome.True, Message = @"requires is valid", PrimaryILOffset = 19, MethodILOffset = 31)]
		    	    [RegressionOutcome(Outcome = ProofOutcome.True, Message = @"assert is valid", PrimaryILOffset = 43, MethodILOffset = 0)]*/
		    public void Use2(@NonNegative int len, @NonNegative int newSize)
	    {
	      //Y: Contract.Requires(newSize >= 0);
	      //YN: Contract.Requires(len > newSize);

	      int[] a = new int[len];

	      int[] call = Arr(a, newSize);
	      // Simplify the postconditions to call.Length == newSize

	      //N: Contract.Assert(call.Length == newSize);
	    }

	    // ok
		    /*[ClousotRegressionTest]
		    	    [RegressionOutcome(Outcome = ProofOutcome.True, Message = @"Array creation : ok", PrimaryILOffset = 22, MethodILOffset = 0)]
		    	    [RegressionOutcome(Outcome = ProofOutcome.True, Message = @"valid non-null reference (as receiver)", PrimaryILOffset = 31, MethodILOffset = 0)]
		    	    [RegressionOutcome(Outcome = ProofOutcome.Top, Message = @"Possible use of a null array 'call'", PrimaryILOffset = 38, MethodILOffset = 0)]
		    	    [RegressionOutcome(Outcome = ProofOutcome.True, Message = @"requires is valid", PrimaryILOffset = 7, MethodILOffset = 31)]
		    	    [RegressionOutcome(Outcome = ProofOutcome.True, Message = @"requires is valid", PrimaryILOffset = 19, MethodILOffset = 31)]
		    	    [RegressionOutcome(Outcome = ProofOutcome.True, Message = @"assert is valid", PrimaryILOffset = 43, MethodILOffset = 0)]*/
		    public void Use3(@NonNegative int len, @NonNegative int newSize)
	    {
	      //Y: Contract.Requires(len >= 0);
	      //YN: Contract.Requires(newSize > len);

	      int[] a = new int[len];

	      int[] call = Arr(a, newSize);
	      // Simplify the postconditions to call.Length == newSize

	      //N: Contract.Assert(call.Length == newSize);
	    }

	    // ok
		    /*[ClousotRegressionTest]
		    	    [RegressionOutcome(Outcome = ProofOutcome.True, Message = @"Array creation : ok", PrimaryILOffset = 25, MethodILOffset = 0)]
		    	    [RegressionOutcome(Outcome = ProofOutcome.True, Message = @"valid non-null reference (as receiver)", PrimaryILOffset = 34, MethodILOffset = 0)]
		    	    [RegressionOutcome(Outcome = ProofOutcome.Top, Message = @"Possible use of a null array 'call'", PrimaryILOffset = 41, MethodILOffset = 0)]
		    	    [RegressionOutcome(Outcome = ProofOutcome.True, Message = @"requires is valid", PrimaryILOffset = 7, MethodILOffset = 34)]
		    	    [RegressionOutcome(Outcome = ProofOutcome.True, Message = @"requires is valid", PrimaryILOffset = 19, MethodILOffset = 34)]
		    	    [RegressionOutcome(Outcome = ProofOutcome.Top, Message = @"assert unproven", PrimaryILOffset = 46, MethodILOffset = 0)]*/
		    public void Use4(@NonNegative int len, @NonNegative int newSize)
	    {
	      //Y: Contract.Requires(len >= 0);
		    	//Y: Contract.Requires(newSize >= 0);

	      int[] a = new int[len];

	      int[] call = Arr(a, newSize);
	      // Cannot simplify the postcondition

	      //N: Contract.Assert(call.Length == newSize);
	    }
	  }
class ForAllTest
{
	
	/*[ClousotRegressionTest]
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
	[RegressionOutcome(Outcome=ProofOutcome.True,Message="valid non-null reference (as receiver)",PrimaryILOffset=78,MethodILOffset=0)]*/
	public int SumLengths(String[] input)
  {
    //N: Contract.Requires(input == null || Contract.ForAll(0, input.Length, j => input[j] != null));

    int c = 0;
    if (input != null)
    {
      for (int i = 0; i < input.length; i++)
      {
        c += input[i].length();
      }
    }

    return c;
  }
}


