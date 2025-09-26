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
import org.checkerframework.checker.index.qual.IndexOrHigh;
import org.checkerframework.checker.index.qual.LTLengthOf;

  class ExamplesFromPapers_Examples
  {

    // bubblesort simplified, from Cousot Halbwachs 1978
    // [ClousotRegressionTest("fast")] // it fails with fast, but we do not really care anymore
    /*[ClousotRegressionTest("simplex")]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = @"Lower bound access ok", PrimaryILOffset = 25, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = @"Upper bound access ok", PrimaryILOffset = 25, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = @"Lower bound access ok", PrimaryILOffset = 28, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = @"Upper bound access ok", PrimaryILOffset = 28, MethodILOffset = 0)]*/
    private void Test5(@IndexOrHigh("#2") int N, int[] K)
    {
      //YN: Contract.Requires(K.Length == N);

    	@IndexOrHigh("K") int b, j, t;
      b = N;
      while (b >= 1)
      {
        j = 1;
        t = 0;
        while (j <= b - 1)
        {
          if (K[j - 1] > K[j])
          {
            // exchange j and j + 1; no side effects
            t = j;
          }
          j++;
        }
        if (t == 0) return;
        b = t;
      }
    }


  }

  class SortAlgorithms
  {
    /*[ClousotRegressionTest("fastnooptions")]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = @"Lower bound access ok", PrimaryILOffset = 20, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = @"Upper bound access ok", PrimaryILOffset = 20, MethodILOffset = 0)]*/
    private void If0(int[] K)
    {
      int b, j;

      b = K.length;

      if (b >= 1)
      {
        j = 1;

        if (j < b)
        {
          K[j - 1] = 12;
        }
      }
    }

    // F: For some reason we need the infoct option to prove it
    /*[ClousotRegressionTest("fastnooptions")]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = @"Lower bound access ok", PrimaryILOffset = 18, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = @"Upper bound access ok", PrimaryILOffset = 18, MethodILOffset = 0)]*/
    private void If1(int[] K)
    {
      @IndexOrHigh("K") int b, j, t;

      b = K.length;

      while (b >= 1)
      {
        j = 1;
        t = 0;

        if (j < b)
        {
          // Contract.Assert(b <= K.Length); // ok
          // Contract.Assert(j < K.Length); // ok
          K[j] = 12; // not ok  

          // Contract.Assert(j -1< K.Length); // not ok

          // K[j - 1] = 12;  // not ok

          j++;
        }

        b = t;
      }
    }

    /*[ClousotRegressionTest("fastnooptions")]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = @"Lower bound access ok", PrimaryILOffset = 18, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = @"Upper bound access ok", PrimaryILOffset = 18, MethodILOffset = 0)]*/
    private void BubbleSort0(int[] K)
    {
      int b, j, t;

      b = K.length;

      while (b >= 1)
      {
        j = 1;
        t = 0;

        while (j < b)
        {
          // Contract.Assert(b <= K.Length);

          K[j - 1] = 12;

          j++;
        }
        if (t == 0)
        {
          return;
        }

        b = t;
      }
    }

    // [ClousotRegressionTest("fast")]
    /*[ClousotRegressionTest("simplex")]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = @"Lower bound access ok", PrimaryILOffset = 16, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = @"Lower bound access ok", PrimaryILOffset = 19, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = @"Lower bound access ok", PrimaryILOffset = 27, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = @"Upper bound access ok", PrimaryILOffset = 27, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = @"Lower bound access ok", PrimaryILOffset = 34, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = @"Upper bound access ok", PrimaryILOffset = 34, MethodILOffset = 0)]
    [RegressionOutcome(Outcome=ProofOutcome.True, Message=@"Upper bound access ok", PrimaryILOffset=16, MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True, Message=@"Upper bound access ok", PrimaryILOffset=19, MethodILOffset=0)]*/
    private void BubbleSort(int[] K)
    {
      @IndexOrHigh("K") int b, j, t;

      b = K.length;

      while (b >= 1)
      {
        j = 1;
        t = 0;

        while (j < b)
        {
          // Contract.Assert(b <= K.Length);

          if (K[j - 1] > K[j])
          {
            Swap(/*ref*/ K[j - 1], /*ref*/ K[j]);

            t = j;
          }

          j++;
        }
        if (t == 0)
        {
          return;
        }

        b = t;
      }
    }

    /*[ClousotRegressionTest("simplex")]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "Lower bound access ok", PrimaryILOffset = 16, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "Upper bound access ok", PrimaryILOffset = 16, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "Lower bound access ok", PrimaryILOffset = 19, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "Upper bound access ok", PrimaryILOffset = 19, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = @"Lower bound access ok", PrimaryILOffset = 26, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = @"Upper bound access ok", PrimaryILOffset = 26, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "Lower bound access ok", PrimaryILOffset = 34, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "Upper bound access ok", PrimaryILOffset = 34, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = @"Upper bound access ok", PrimaryILOffset = 35, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = @"Lower bound access ok", PrimaryILOffset = 35, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = @"Lower bound access ok", PrimaryILOffset = 39, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = @"Upper bound access ok", PrimaryILOffset = 39, MethodILOffset = 0)]*/
    private void BubbleSortSimplex(int[] K)
    {
    	@IndexOrHigh("K") int b, j, t;

      b = K.length;

      while (b >= 1)
      {
        j = 1;
        t = 0;

        while (j < b)
        {
          if (K[j - 1] > K[j])
          {
            int tmp = K[j - 1];
            K[j - 1] = K[j];
            K[j] = tmp;
            
            t = j;
          }

          j++;
        }
        if (t == 0)
        {
          return;
        }

        b = t;
      }
    }

    private <T>void Swap(/*ref*/ T x, /*ref*/ T y)
    {
      //N: Contract.Ensures(x.Equals(Contract.OldValue(y)));
      //N: Contract.Ensures(y.Equals(Contract.OldValue(x)));

      T tmp = x;
      x = y;
      y = tmp;
    }
  }

class __{

   
    /*[ClousotRegressionTest("simplex")]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"Array creation : ok",PrimaryILOffset=5,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"Lower bound access ok",PrimaryILOffset=21,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"Upper bound access ok",PrimaryILOffset=21,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"Lower bound access ok",PrimaryILOffset=26,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"Upper bound access ok",PrimaryILOffset=26,MethodILOffset=0)]*/
    public static String[] Test7(String[] x)
    {
      int max = 0;
      String[] result = new String[x.length];

      for (String s : x)
      {
        // proven by infoct (which infers max < x.Length)
        result[max] = s;

        if (s == null)
        {
          max++;
        }
      }

      return result;
    }


    /*[ClousotRegressionTest]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = @"Lower bound access ok", PrimaryILOffset = 23, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = @"Upper bound access ok", PrimaryILOffset = 23, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = @"Lower bound access ok", PrimaryILOffset = 40, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = @"Upper bound access ok", PrimaryILOffset = 40, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.Top, Message = @"assert unproven", PrimaryILOffset = 47, MethodILOffset = 0)]*/
    public void TestCausingABugInRemovingRedundancies(String[] myStrings)
    {
      //N: Contract.Requires(myStrings != null);
      for (int i = 0; i < myStrings.length; i++)
      {
        myStrings[i] = "a";
      }

      for (int i = 0; i < myStrings.length; i++)
      {
        //N: Contract.Assert(myStrings[i] != null);  // Needs -arrays
      }
    }

    /*[ClousotRegressionTest("simplex")]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = @"Lower bound access ok", PrimaryILOffset = 20, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = @"Upper bound access ok", PrimaryILOffset = 20, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = @"assert is valid", PrimaryILOffset = 29, MethodILOffset = 0)]*/
    public static void Partition(int[] arr)
    {
      //N: Contract.Requires(arr != null);

      int z = 0;

      for (int i = 0; i < arr.length; i++)
      {
        //Contract.Assert(z <= i); // This is inferred by Subpolyhedra
        if (arr[i] == 0)
        {
          //Y: Contract.Assert(z < arr.Length);
        	@LTLengthOf("arr") int _1 = z; 
          z++;
        }
      }
    }
  }

