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

import org.checkerframework.checker.index.qual.LTLengthOf;
import org.checkerframework.checker.index.qual.NonNegative;
import org.checkerframework.checker.index.qual.Positive;

  class SymbolicForAll_Example
  {
    /*[ClousotRegressionTest]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"Array creation : ok",PrimaryILOffset=10,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"Lower bound access ok",PrimaryILOffset=19,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"Upper bound access ok",PrimaryILOffset=19,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"Lower bound access ok",PrimaryILOffset=24,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"Upper bound access ok",PrimaryILOffset=24,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"Lower bound access ok",PrimaryILOffset=27,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"Upper bound access ok",PrimaryILOffset=27,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"valid non-null reference (as array)",PrimaryILOffset=19,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"valid non-null reference (as array)",PrimaryILOffset=24,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"valid non-null reference (as array)",PrimaryILOffset=27,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"assert is valid",PrimaryILOffset=31,MethodILOffset=0)]*/
    public int[] Example0(int x, int y)
    {
      //N: Contract.Requires(x < y);
      
      int[] result = new int[2];
      
      result[0] = x;
      result[1] = 11; // Adding it to fool the heap analysis so that it does not remeber the last write to result
      
      //N: Contract.Assert(result[0] < y);
      
      return result;
    }

    /*[ClousotRegressionTest]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"Array creation : ok",PrimaryILOffset=33,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"Lower bound access ok",PrimaryILOffset=46,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"Upper bound access ok",PrimaryILOffset=46,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"valid non-null reference (as field receiver)",PrimaryILOffset=8,MethodILOffset=0)]
#if !CLOUSOT2
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"valid non-null reference (as field receiver)",PrimaryILOffset=14,MethodILOffset=0)] 
#endif
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"valid non-null reference (as field receiver)",PrimaryILOffset=8,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"valid non-null reference (as field receiver)",PrimaryILOffset=28,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"valid non-null reference (as field receiver)",PrimaryILOffset=53,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"valid non-null reference (as array)",PrimaryILOffset=46,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"assert is valid",PrimaryILOffset=95,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"assert is valid",PrimaryILOffset=118,MethodILOffset=0)]

#if NETFRAMEWORK_4_0
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"requires is valid",PrimaryILOffset=17,MethodILOffset=90)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"requires is valid",PrimaryILOffset=39,MethodILOffset=90)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"requires is valid",PrimaryILOffset=17,MethodILOffset=113)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"requires is valid",PrimaryILOffset=39,MethodILOffset=113)]
#else
	#if CLOUSOT2
		[RegressionOutcome(Outcome=ProofOutcome.True,Message="requires is valid",PrimaryILOffset=3,MethodILOffset=90)]
		[RegressionOutcome(Outcome=ProofOutcome.True,Message="requires is valid",PrimaryILOffset=25,MethodILOffset=90)]
		[RegressionOutcome(Outcome=ProofOutcome.True,Message="requires is valid",PrimaryILOffset=3,MethodILOffset=113)]
		[RegressionOutcome(Outcome=ProofOutcome.True,Message="requires is valid",PrimaryILOffset=25,MethodILOffset=113)]
	#else
		[RegressionOutcome(Outcome=ProofOutcome.True,Message=@"requires is valid",PrimaryILOffset=22,MethodILOffset=113)]
		[RegressionOutcome(Outcome=ProofOutcome.True,Message=@"requires is valid",PrimaryILOffset=44,MethodILOffset=113)]
		[RegressionOutcome(Outcome=ProofOutcome.True,Message=@"requires is valid",PrimaryILOffset=22,MethodILOffset=90)]
		[RegressionOutcome(Outcome=ProofOutcome.True,Message=@"requires is valid",PrimaryILOffset=44,MethodILOffset=90)]
	#endif
#endif*/
    public int[] Example1(@Positive int x)
    {
      //Y: Contract.Requires(x > 0);
      @NonNegative int[] result = new int[x];

      for (int i = 0; i < x; i++)
      {
        result[i] = i;
      }

      //Y: Contract.Assert(Contract.ForAll(result, el => el >= 0));
      @NonNegative int [] _1 = result; 
      //N: Contract.Assert(Contract.ForAll(result, el => el < x));

      return result;
    }
/*
    [ClousotRegressionTest]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"Array creation : ok",PrimaryILOffset=41,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"Lower bound access ok",PrimaryILOffset=54,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"Upper bound access ok",PrimaryILOffset=54,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"valid non-null reference (as field receiver)",PrimaryILOffset=8,MethodILOffset=0)]
#if !CLOUSOT2
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"valid non-null reference (as field receiver)",PrimaryILOffset=27,MethodILOffset=0)] 
#endif
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"valid non-null reference (as field receiver)",PrimaryILOffset=8,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"valid non-null reference (as array)",PrimaryILOffset=61,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"valid non-null reference (as array)",PrimaryILOffset=54,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"assert is valid",PrimaryILOffset=83,MethodILOffset=0)]
#if NETFRAMEWORK_4_0
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"requires is valid",PrimaryILOffset=17,MethodILOffset=78)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"requires is valid",PrimaryILOffset=39,MethodILOffset=78)]
#else
	#if CLOUSOT2
		[RegressionOutcome(Outcome=ProofOutcome.True,Message=@"requires is valid",PrimaryILOffset=3,MethodILOffset=78)]
		[RegressionOutcome(Outcome=ProofOutcome.True,Message=@"requires is valid",PrimaryILOffset=25,MethodILOffset=78)]
	#else
		[RegressionOutcome(Outcome=ProofOutcome.True,Message=@"requires is valid",PrimaryILOffset=22,MethodILOffset=78)]
		[RegressionOutcome(Outcome=ProofOutcome.True,Message=@"requires is valid",PrimaryILOffset=44,MethodILOffset=78)]
	#endif
#endif*/
    public int[] Example2(@NonNegative int z, int Sup)
    {
      //Y: Contract.Requires(z >= 0);
      //N: Contract.Requires(z < Sup);

      int[] r = new int[100];
      for (int i = 0; i < r.length; i++)
      {
        r[i] = z;
      }

      //N: Contract.Assert(Contract.ForAll(r, el => el < Sup));

      return r;
    }

    /*[ClousotRegressionTest]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"Array creation : ok",PrimaryILOffset=20,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"Lower bound access ok",PrimaryILOffset=42,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"Upper bound access ok",PrimaryILOffset=42,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"valid non-null reference (as field receiver)",PrimaryILOffset=25,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"valid non-null reference (as field receiver)",PrimaryILOffset=49,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"valid non-null reference (as array)",PrimaryILOffset=54,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"valid non-null reference (as field receiver)",PrimaryILOffset=35,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"valid non-null reference (as array)",PrimaryILOffset=42,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"valid non-null reference (as field receiver)",PrimaryILOffset=59,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"valid non-null reference (as field receiver)",PrimaryILOffset=87,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"assert is valid",PrimaryILOffset=81,MethodILOffset=0)]
#if NETFRAMEWORK_4_0
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"requires is valid",PrimaryILOffset=17,MethodILOffset=76)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"requires is valid",PrimaryILOffset=39,MethodILOffset=76)]
#else
	#if CLOUSOT2
		[RegressionOutcome(Outcome=ProofOutcome.True,Message=@"requires is valid",PrimaryILOffset=3,MethodILOffset=76)]
		[RegressionOutcome(Outcome=ProofOutcome.True,Message=@"requires is valid",PrimaryILOffset=25,MethodILOffset=76)]
	#else
		[RegressionOutcome(Outcome=ProofOutcome.True,Message=@"requires is valid",PrimaryILOffset=22,MethodILOffset=76)]
		[RegressionOutcome(Outcome=ProofOutcome.True,Message=@"requires is valid",PrimaryILOffset=44,MethodILOffset=76)]
	#endif
#endif*/
    public int[] Example3(@NonNegative int z)
    {
      //Y: Contract.Requires(z >= 0);

    	@LTLengthOf("r") int[] r = new int[z];
      for (int i = 0; i < r.length; i++)
      {
        r[i] = i;
      }

      //Y: Contract.Assert(Contract.ForAll(r, el => el < r.Length));
      @LTLengthOf("r") int [] _1 = r; 

      return r;
    }

  }
 // TODO : Understand why it does not work. 
  abstract  class SystemWeb_Example
  {
    public boolean[] Items;

    /*[ClousotRegressionTest]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"Array creation : ok",PrimaryILOffset=3,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"Lower bound access ok",PrimaryILOffset=20,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"Upper bound access ok",PrimaryILOffset=20,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"Array creation : ok",PrimaryILOffset=32,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"Lower bound access ok",PrimaryILOffset=55,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"Upper bound access ok",PrimaryILOffset=55,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"Array creation : ok",PrimaryILOffset=95,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"valid non-null reference (as field receiver)",PrimaryILOffset=62,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.Top,Message=@"Possible use of a null array 'this.Items'",PrimaryILOffset=67,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"valid non-null reference (as field receiver)",PrimaryILOffset=14,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"valid non-null reference (as array)",PrimaryILOffset=20,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"valid non-null reference (as array)",PrimaryILOffset=25,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"valid non-null reference (as receiver)",PrimaryILOffset=41,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"valid non-null reference (as array)",PrimaryILOffset=55,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"requires is valid",PrimaryILOffset=13,MethodILOffset=41)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"requires is valid",PrimaryILOffset=33,MethodILOffset=41)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"requires is valid",PrimaryILOffset=51,MethodILOffset=41)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"requires is valid",PrimaryILOffset=81,MethodILOffset=41)]
    [RegressionOutcome(Outcome=ProofOutcome.Top,Message=@"assert unproven",PrimaryILOffset=89,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"requires is valid",PrimaryILOffset=13,MethodILOffset=108)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"requires is valid",PrimaryILOffset=31,MethodILOffset=108)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"requires is valid",PrimaryILOffset=55,MethodILOffset=108)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"requires is valid",PrimaryILOffset=79,MethodILOffset=108)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"requires is valid",PrimaryILOffset=98,MethodILOffset=108)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"requires is valid",PrimaryILOffset=132,MethodILOffset=108)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"requires is valid",PrimaryILOffset=166,MethodILOffset=108)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"assert is valid",PrimaryILOffset=149,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.Top,Message=@"assert unproven",PrimaryILOffset=173,MethodILOffset=0)]
#if NETFRAMEWORK_4_0
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"requires is valid",PrimaryILOffset=17,MethodILOffset=84)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"requires is valid",PrimaryILOffset=39,MethodILOffset=84)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"requires is valid",PrimaryILOffset=17,MethodILOffset=144)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"requires is valid",PrimaryILOffset=39,MethodILOffset=144)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"requires is valid",PrimaryILOffset=17,MethodILOffset=168)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"requires is valid",PrimaryILOffset=39,MethodILOffset=168)]
#else
	#if CLOUSOT2
		[RegressionOutcome(Outcome=ProofOutcome.True,Message=@"requires is valid",PrimaryILOffset=3,MethodILOffset=84)]
		[RegressionOutcome(Outcome=ProofOutcome.True,Message=@"requires is valid",PrimaryILOffset=25,MethodILOffset=84)]
		[RegressionOutcome(Outcome=ProofOutcome.True,Message=@"requires is valid",PrimaryILOffset=3,MethodILOffset=168)]
		[RegressionOutcome(Outcome=ProofOutcome.True,Message=@"requires is valid",PrimaryILOffset=25,MethodILOffset=168)]
		[RegressionOutcome(Outcome=ProofOutcome.True,Message=@"requires is valid",PrimaryILOffset=3,MethodILOffset=144)]
		[RegressionOutcome(Outcome=ProofOutcome.True,Message=@"requires is valid",PrimaryILOffset=25,MethodILOffset=144)]
	#else
	    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"requires is valid",PrimaryILOffset=22,MethodILOffset=84)]
		[RegressionOutcome(Outcome=ProofOutcome.True,Message=@"requires is valid",PrimaryILOffset=44,MethodILOffset=84)]
		[RegressionOutcome(Outcome=ProofOutcome.True,Message=@"requires is valid",PrimaryILOffset=22,MethodILOffset=168)]
		[RegressionOutcome(Outcome=ProofOutcome.True,Message=@"requires is valid",PrimaryILOffset=44,MethodILOffset=168)]
		[RegressionOutcome(Outcome=ProofOutcome.True,Message=@"requires is valid",PrimaryILOffset=22,MethodILOffset=144)]
		[RegressionOutcome(Outcome=ProofOutcome.True,Message=@"requires is valid",PrimaryILOffset=44,MethodILOffset=144)]
	#endif
#endif*/
    public int[] get_SelectedIndices()
    {
      int length = 0;
      int[] sourceArray = new int[3];
      for (int i = 0; i < this.Items.length; i++)
      {
        if (this.Items[i])
        {
          if (length == sourceArray.length)
          {
            int[] array = new int[length + length];
            System.arraycopy(sourceArray, 0, array, 0, sourceArray.length);
            //sourceArray.CopyTo(array, 0);
            sourceArray = array;
          }
          sourceArray[length++] = i;
        }
      }
      //Y: (UNPROVEN) Contract.Assert(Contract.ForAll(sourceArray, e => e < this.Items.Length));
   // :: error: (assignment.type.incompatible)
      @LTLengthOf("this.Items") int [] _3 = sourceArray;

      @NonNegative int[] destinationArray = new int[length];
      System.arraycopy(sourceArray, 0, destinationArray, 0, length);

      //Y: Contract.Assert(Contract.ForAll(destinationArray, e => e >= 0));
      @NonNegative int [] _1 = destinationArray;
      //Y: (UNPROVEN) Contract.Assert(Contract.ForAll(destinationArray, e => e < this.Items.Length));
      // :: error: (assignment.type.incompatible)
      @LTLengthOf("this.Items") int [] _2 = destinationArray; 

      return destinationArray;
    }
  }
