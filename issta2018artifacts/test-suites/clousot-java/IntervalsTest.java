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

import org.checkerframework.checker.index.qual.NonNegative;
import org.checkerframework.common.value.qual.ArrayLen;
import org.checkerframework.common.value.qual.IntRange;

  class TestIntervals_DateTime
  {
	  private static int @ArrayLen(13)[] DaysToMonth365, DaysToMonth366;

	    static
	    {
	      DaysToMonth365 = new int[] { 0, 0x1f, 0x3b, 90, 120, 0x97, 0xb5, 0xd4, 0xf3, 0x111, 0x130, 0x14e, 0x16d };
	      DaysToMonth366 = new int[] { 0, 0x1f, 60, 0x5b, 0x79, 0x98, 0xb6, 0xd5, 0xf4, 0x112, 0x131, 0x14f, 0x16e };
	    }
	    
	    long dateData;

	    private @NonNegative long get_InternalTicks()
	    {
	      // ok
	      /*[ClousotRegressionTest]
	      [RegressionOutcome(Outcome=ProofOutcome.True,Message="ensures is valid",PrimaryILOffset=12,MethodILOffset=33)]*/
	     
	        //Y: Contract.Ensures(Contract.Result<long>() >= 0);
	        return (((long)this.dateData) & 0x3fffffffffffffffL);
	    }
	    
    // All the division by zero should be ok
    // 3 (out of 4) array accesses should raise a warning
    // One assertion fails because of lack of object invariant
    /*[RegressionOutcome(Outcome=ProofOutcome.Top,Message="Array access might be below the lower bound",PrimaryILOffset=204,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.Top,Message="Array access might be above the upper bound",PrimaryILOffset=204,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.Top,Message="Array access might be below the lower bound",PrimaryILOffset=221,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message="Upper bound access ok",PrimaryILOffset=221,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message="Division by zero ok",PrimaryILOffset=21,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message="Division by zero ok",PrimaryILOffset=30,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message="Division by zero ok",PrimaryILOffset=48,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message="Division by zero ok",PrimaryILOffset=72,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message="Division by zero ok",PrimaryILOffset=90,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message="assert is valid",PrimaryILOffset=1,MethodILOffset=0)]
    [RegressionOutcome(Outcome = ProofOutcome.Top, Message = "assert unproven", PrimaryILOffset = 181, MethodILOffset = 0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"No overflow",PrimaryILOffset=21,MethodILOffset=0)]
[RegressionOutcome(Outcome=ProofOutcome.True,Message=@"No overflow",PrimaryILOffset=30,MethodILOffset=0)]
[RegressionOutcome(Outcome=ProofOutcome.True,Message=@"No overflow",PrimaryILOffset=48,MethodILOffset=0)]
[RegressionOutcome(Outcome=ProofOutcome.True,Message=@"No overflow",PrimaryILOffset=72,MethodILOffset=0)]
 [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"No overflow",PrimaryILOffset=90,MethodILOffset=0)]
    [ClousotRegressionTest]*/  
    private int GetDatePart(int part)
    {
      //N: Contract.Assert(0xc92a69c000L != 0);    // This is converted by the compiler to "1", and it undercovered a bug in the CheckIfHolds of intervals, so we keep it

    
    	
      int num2 = (int)(this.get_InternalTicks() / 0xc92a69c000L);
      int num3 = num2 / 0x23ab1;
      num2 -= num3 * 0x23ab1;
      int num4 = num2 / 0x8eac;
      if (num4 == 4)
      {
        num4 = 3;
      }
      num2 -= num4 * 0x8eac;
      int num5 = num2 / 0x5b5;
      num2 -= num5 * 0x5b5;
      int num6 = num2 / 0x16d;

      if (num6 == 4)
      {
        num6 = 3;
      }
      if (part == 0)
      {
        return (((((num3 * 400) + (num4 * 100)) + (num5 * 4)) + num6) + 1);
      }
      num2 -= num6 * 0x16d;
      if (part == 1)
      {
        return (num2 + 1);
      }

      int[] numArray = ((num6 == 3) && ((num5 != 0x18) || (num4 == 3))) ? DaysToMonth366 : DaysToMonth365;

      // TODO: With class invariants should infer DaysToMonth35?.Length == 13
      //Y: Contract.Assert(numArray.Length == 13);
      int @ArrayLen(13)[] _1 = numArray;

      int index = num2 >> 6;
      
      // Here should prove index < 13, which is quite hard
      // :: error: (array.access.unsafe.low) :: error: (array.access.unsafe.high.range)
      while (num2 >= numArray[index])
      {
        index++;
      }
      if (part == 2)
      {
        return index;
      }
      // :: error: (array.access.unsafe.low) :: error: (array.access.unsafe.high.range)
      return ((num2 - numArray[index - 1]) + 1);
    } 
  }

  class TestIntervals_Enum
  {
   /* [ClousotRegressionTest]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "Array creation : ok", PrimaryILOffset = 3, MethodILOffset = 0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"No overflow (caused by a negative array size)",PrimaryILOffset=3,MethodILOffset=0)]
#if NETFRAMEWORK_4_0
    [RegressionOutcome(Outcome=ProofOutcome.True,Message="Lower bound access ok",PrimaryILOffset=28,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.Top,Message="Array access might be above the upper bound",PrimaryILOffset=28,MethodILOffset=0)]
#else
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "Lower bound access ok", PrimaryILOffset = 32, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.Top, Message = "Array access might be above the upper bound", PrimaryILOffset = 32, MethodILOffset = 0)]
#endif*/
    private static Object GetHashEntry(Class<?> enumType, boolean[] flds, /*u*/long val)
    {
      /*u*/long[] values = new /*u*/long[flds.length];
      String[] names = new String[flds.length]; // /optimize+ removes this array creation,

      for (int i = 1; i < values.length; i++)
      {
        int j = i;

        while (values[j - 1] > val)
        {
          j--;

          if (j == 0)
          {
            break;
          }
        }
      }
      return null;
    }
  }

  class Repro_AlexeyR
  {
    /*[ClousotRegressionTest]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"Array creation : ok",PrimaryILOffset=1,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"Lower bound access ok",PrimaryILOffset=9,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"Upper bound access ok",PrimaryILOffset=9,MethodILOffset=0)]	
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"assert is valid",PrimaryILOffset=22,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"No overflow (caused by a negative array size)",PrimaryILOffset=1,MethodILOffset=0)]*/
    public void ldelem()
    {
      byte[] buf = new byte[1];
      byte val = buf[0];
      //Y: Contract.Assert(val <= byte.MaxValue);
      @IntRange(to = Byte.MAX_VALUE) int _1 = val;
    }
  }
  