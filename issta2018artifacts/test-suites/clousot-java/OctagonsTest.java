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
import org.checkerframework.checker.index.qual.IndexFor;
import org.checkerframework.checker.index.qual.IndexOrHigh;
import org.checkerframework.checker.index.qual.LTLengthOf;
import org.checkerframework.checker.index.qual.NonNegative;

 class TestOctagons_BasicTests
  {
    // #2 ok
    /*[ClousotRegressionTest]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = @"Array creation : ok", PrimaryILOffset = 13, MethodILOffset = 0)]*/
    public boolean[] M2(@NonNegative int x)
    {
      //Y: Contract.Requires(x >= 0);

      return new boolean[x];
    }

    // #5 ok
    /*[ClousotRegressionTest]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = @"Lower bound access ok", PrimaryILOffset = 25, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = @"Upper bound access ok", PrimaryILOffset = 25, MethodILOffset = 0)]*/
    public boolean M5(boolean[] a,@IndexFor("#1") int index)
    {
    	//Y: Contract.Requires(index >= 0);
    	//Y: Contract.Requires(index < a.Length);

      return a[index];
    }


    //#8 TODO
    /*[ClousotRegressionTest]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = @"Lower bound access ok", PrimaryILOffset = 22, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = @"Upper bound access ok", PrimaryILOffset = 22, MethodILOffset = 0)]*/
    public void M8(int[] array)
    {
      int index = 0;
      int length = array.length;

      if (index < length)
      {
        index++;
      }

      if (index == length)
        return;

      array[index] = 0;
    }

  }


  class Loops
  {
    /*[ClousotRegressionTest]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"Lower bound access ok",PrimaryILOffset=8,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"Upper bound access ok",PrimaryILOffset=8,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"assert is valid",PrimaryILOffset=16,MethodILOffset=0)]*/
    public static void Loop0(String[] x)
    {
      int max = 0;
      for (int i = 0; i < x.length; i++)
      {
        String z = x[i];

        //Y: Contract.Assert(max < x.Length);
        @LTLengthOf("x") int _1= max;

        if (z == null)
        {
          max++;
        }
      }
    }
  }




