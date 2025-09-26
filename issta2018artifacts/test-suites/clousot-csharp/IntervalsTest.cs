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

// Original file: Microsoft.Research\RegressionTest\ClousotTests\Sources\TestIntervals\IntervalsTest.cs
// Modified for Index Checker evaluation: removed parts that are not representable or are irrelevant to array bounds

using System.Diagnostics.Contracts;
using Microsoft.Research.ClousotRegression;
using System;

namespace TestIntervals
{
  class IntervalsBasic
  {
    [ClousotRegressionTest]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "assert is valid", PrimaryILOffset = 32, MethodILOffset = 0)]
    public void M0(int x, int z)
    {
      //  Contract.Requires(x + z== 5);
      Contract.Requires(x >= 0);
      Contract.Requires(x >= 12);

      Contract.Assert(x >= 0);
    }

    // two warnings
    // We check twice the postcondition, as /optimize+ generates two "ret" instructions
    [ClousotRegressionTest]
    [RegressionOutcome(Outcome = ProofOutcome.Top, Message = "Possible negation of MinValue of type Int32. The static checker determined that the condition 'x != Int32.MinValue' should hold on entry. Nevertheless, the condition may be too strong for the callers. If you think it is ok, add a precondition to document it: Contract.Requires(x != Int32.MinValue);", PrimaryILOffset = 18, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "ensures is valid", PrimaryILOffset = 8, MethodILOffset = 19)]
    [RegressionOutcome(Outcome = ProofOutcome.Top, Message = @"ensures unproven: Contract.Result<int>() > 0. The static checker determined that the condition 'x > 0' should hold on entry. Nevertheless, the condition may be too strong for the callers. If you think it is ok, add a precondition to document it: Contract.Requires(x > 0);. Is it an off-by-one? The static checker can prove x > (0 - 1) instead", PrimaryILOffset = 8, MethodILOffset = 21)]
    public int Abs0(int x)
    {
      Contract.Ensures(Contract.Result<int>() > 0);

      if (x < 0)
        return -x;
      else
        return x;
    }

    // Ok
    // We check twice the postcondition, as /optimize+ generates two "ret" instructions
    [ClousotRegressionTest]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "ensures is valid", PrimaryILOffset = 11, MethodILOffset = 22)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "ensures is valid", PrimaryILOffset = 11, MethodILOffset = 24)]
    [RegressionOutcome(Outcome = ProofOutcome.Top, Message = "Possible negation of MinValue of type Int32. The static checker determined that the condition 'x != Int32.MinValue' should hold on entry. Nevertheless, the condition may be too strong for the callers. If you think it is ok, add a precondition to document it: Contract.Requires(x != Int32.MinValue);", PrimaryILOffset = 21, MethodILOffset = 0)]
    public int Abs1(int x)
    {
      Contract.Ensures(Contract.Result<int>() >= 0);

      if (x < 0)
        return -x;
      else
        return x;
    }

    // ok
    [ClousotRegressionTest]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "ensures is valid", PrimaryILOffset = 9, MethodILOffset = 29)]
    public int Arithmetics0()
    {
      Contract.Ensures(Contract.Result<int>() == -6);
      int y, z, x, t;

      y = 1;
      z = 16;
      x = 9;

      t = y - z + x;

      return t;
    }

    [ClousotRegressionTest]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"ensures is valid",PrimaryILOffset=21,MethodILOffset=33)]
    public int Arithmetics1_Overflow(int x)
    {
      Contract.Requires(x == 1);
      Contract.Ensures(Contract.Result<int>() == Int32.MinValue);

      return Int32.MaxValue + x; // We know it will overflow
    }

    [ClousotRegressionTest]
//    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"No underflow",PrimaryILOffset=32,MethodILOffset=0)]
//    [RegressionOutcome(Outcome=ProofOutcome.False,Message=@"Overflow in the arithmetic operation",PrimaryILOffset=32,MethodILOffset=0)]
//    [RegressionOutcome(Outcome=ProofOutcome.False,Message=@"ensures is false: Contract.Result<int>() == Int32.MinValue",PrimaryILOffset=21,MethodILOffset=33)]

    // F: a bug fixing in the conversion in IEEE754 intervals introduced a loss of precision, so we only say "top", instead of getting the definite false 
    [RegressionOutcome(Outcome=ProofOutcome.Top,Message="Possible underflow in the arithmetic operation",PrimaryILOffset=32,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.Top,Message="Possible overflow in the arithmetic operation",PrimaryILOffset=32,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.Top,Message="ensures unproven: Contract.Result<int>() == Int32.MinValue",PrimaryILOffset=21,MethodILOffset=33)]
    public int Arithmetics2_CheckedOverlow(int x)
    {
      Contract.Requires(x == 1);
      Contract.Ensures(Contract.Result<int>() == Int32.MinValue);

      return checked(Int32.MaxValue + x); 
    }

    [ClousotRegressionTest]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = @"Division by zero ok", PrimaryILOffset = 18, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = @"Division by zero ok", PrimaryILOffset = 23, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = @"Division by zero ok", PrimaryILOffset = 29, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.Top, Message = @"ensures unproven: Contract.Result<int>() >= 0. The static checker determined that the condition '(i / 2) >= 0' should hold on entry. Nevertheless, the condition may be too strong for the callers. If you think it is ok, add a precondition to document it: Contract.Requires((i / 2) >= 0);", PrimaryILOffset = 11, MethodILOffset = 24)]
    [RegressionOutcome(Outcome = ProofOutcome.Top, Message = @"ensures unproven: Contract.Result<int>() >= 0. The static checker determined that the condition '((i - 1) / 2) >= 0' should hold on entry. Nevertheless, the condition may be too strong for the callers. If you think it is ok, add a precondition to document it: Contract.Requires(((i - 1) / 2) >= 0);", PrimaryILOffset = 11, MethodILOffset = 30)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = @"No overflow", PrimaryILOffset = 23, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = @"No overflow", PrimaryILOffset = 29, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = @"No overflow", PrimaryILOffset = 18, MethodILOffset = 0)]
    public int Rem_Wrong(int i)
    {
      Contract.Ensures(Contract.Result<int>() >= 0);

      if (i % 2 == 0)
        return i / 2;
      else
        return (i - 1) / 2;
    }

    [ClousotRegressionTest]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = @"Division by zero ok", PrimaryILOffset = 30, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = @"Division by zero ok", PrimaryILOffset = 35, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = @"Division by zero ok", PrimaryILOffset = 41, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = @"ensures is valid", PrimaryILOffset = 23, MethodILOffset = 36)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = @"ensures is valid", PrimaryILOffset = 23, MethodILOffset = 42)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = @"No overflow", PrimaryILOffset = 35, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = @"No overflow", PrimaryILOffset = 41, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = @"No overflow", PrimaryILOffset = 30, MethodILOffset = 0)]
    public int Rem_Ok(int i)
    {
      Contract.Requires(i >= 0);
      Contract.Ensures(Contract.Result<int>() >= 0);

      if (i % 2 == 0)
        return i / 2;
      else
        return (i - 1) / 2;
    }

    [ClousotRegressionTest]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = @"Division by zero ok", PrimaryILOffset = 33, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.Top, Message = @"ensures unproven: Contract.Result<int>() > 0. Is it an off-by-one? The static checker can prove ((value % divisor)) > (0 - 1) instead", PrimaryILOffset = 26, MethodILOffset = 34)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = @"No overflow", PrimaryILOffset = 33, MethodILOffset = 0)]
    public int Rem_PosPos_Wrong(int value, int divisor)
    {
      Contract.Requires(value > 0);
      Contract.Requires(divisor > 0);

      Contract.Ensures(Contract.Result<int>() > 0);

      return value % divisor;
    }

    [ClousotRegressionTest]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = @"Division by zero ok", PrimaryILOffset = 36, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = @"ensures is valid", PrimaryILOffset = 29, MethodILOffset = 37)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = @"No overflow", PrimaryILOffset = 36, MethodILOffset = 0)]
    public int Rem_PosPos_Ok(int value, int divisor)
    {
      Contract.Requires(value > 0);
      Contract.Requires(divisor > 0);

      Contract.Ensures(Contract.Result<int>() >= 0);

      return value % divisor;
    }

    [ClousotRegressionTest]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = @"Division by zero ok", PrimaryILOffset = 33, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.Top, Message = @"ensures unproven: Contract.Result<int>() < 0", PrimaryILOffset = 26, MethodILOffset = 34)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = @"No overflow", PrimaryILOffset = 33, MethodILOffset = 0)]
    public int Rem_NegPos_Wrong(int value, int divisor)
    {
      Contract.Requires(value < 0);
      Contract.Requires(divisor > 0);

      Contract.Ensures(Contract.Result<int>() < 0);

      return value % divisor;
    }

    [ClousotRegressionTest]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = @"Division by zero ok", PrimaryILOffset = 36, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = @"ensures is valid", PrimaryILOffset = 29, MethodILOffset = 37)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = @"No overflow", PrimaryILOffset = 36, MethodILOffset = 0)]
    public int Rem_NegPos_Ok(int value, int divisor)
    {
      Contract.Requires(value < 0);
      Contract.Requires(divisor > 0);

      Contract.Ensures(Contract.Result<int>() <= 0);

      return value % divisor;
    }

    [ClousotRegressionTest]
    [RegressionOutcome(Outcome = ProofOutcome.Top, Message = @"assert unproven. Is it an off-by-one? The static checker can prove x >= (0 - 1) instead", PrimaryILOffset = 20, MethodILOffset = 0)]
    public void AddMaxValue_Wrong(int x)
    {
      if (x < 0)
      {
        x += 0x7fffffff;
      }

      Contract.Assert(x >= 0);
    }

    [ClousotRegressionTest]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = @"assert is valid", PrimaryILOffset = 20, MethodILOffset = 0)]
    public void AddMaxValue_ok(int x)
    {
      if (x < 0)
      {
        x += 0x7fffffff;
      }

      Contract.Assert(x >= -1);
    }
  }

  class UseGuards
  {
    // ok
    [ClousotRegressionTest]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "ensures is valid", PrimaryILOffset = 11, MethodILOffset = 36)]
    public int Guard0(int k)
    {
      Contract.Ensures(Contract.Result<int>() >= 5);

      int z;

      if (k > 0)
        z = 2 * k + 5;
      else
        z = -2 * k + 5;

      return z;
    }

    // ok 
    [ClousotRegressionTest]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "assert is valid", PrimaryILOffset = 24, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "assert is valid", PrimaryILOffset = 37, MethodILOffset = 0)]
    public int Guard1(bool b)
    {

      int z = 25;

      if (b)
      {
        z = z - 1;
      }
      else
      {
        z = z + 1;
      }

      Contract.Assert(z >= 24);
      Contract.Assert(z <= 26);
      return z; 
    }

    // ok
    [ClousotRegressionTest]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "assert is valid", PrimaryILOffset = 19, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "assert is valid", PrimaryILOffset = 32, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "assert is valid", PrimaryILOffset = 49, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "assert is valid", PrimaryILOffset = 62, MethodILOffset = 0)]
    public int Guard2(bool b)
    {
      int x;

      if (b)
        x = 100;
      else
        x = -100;

      Contract.Assert(-100<= x);
      Contract.Assert(x <= 100); 
      
      x = x + 1;

      Contract.Assert(-99<= x);
      Contract.Assert(x <= 101); 

      return x;
    }

    // ok
    [ClousotRegressionTest]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "assert is valid", PrimaryILOffset = 21, MethodILOffset = 0)]
    public int Guard3()
    {
      int i = 12;
      int r;

      if (i != 56)
        r = 22;
      else
        r = 123;

      Contract.Assert(r == 22);
      return r;
    }

    // ok
    [ClousotRegressionTest]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "assert is valid", PrimaryILOffset = 13, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "assert is valid", PrimaryILOffset = 28, MethodILOffset = 0)]
    public bool Guard4(int i)
    {
      if (i < -5)
      {
        Contract.Assert(i <= -6);
        return true;
      }
      else
      {
        Contract.Assert(i >= -5);
        return false;
      }
    }
  }

  class Loops
  {
    // ok
    [ClousotRegressionTest]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "assert is valid", PrimaryILOffset = 21, MethodILOffset = 0)]
    public void Loop0()
    {
      int i = 0;
      while (i < 100)
        i++;
     
      Contract.Assert(i >= 100);
    }

    // One warning 
    [ClousotRegressionTest]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "assert is valid", PrimaryILOffset = 18, MethodILOffset = 0)]
    public void Loop1()
    {
      int i = 0;
      while (i < 100)
        i++;
      
      Contract.Assert(i == 100); // Proven using widening with thresholds
    }

    // ok
    [ClousotRegressionTest]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "assert is valid", PrimaryILOffset = 34, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "assert is valid", PrimaryILOffset = 13, MethodILOffset = 0)]
    public void Loop2()
    {
      int i = 100;
      while (i > 0)
      {
        Contract.Assert(i <= 100);
        i--;
      }

      Contract.Assert(i <= 100);
    }

    // ok
    [ClousotRegressionTest]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "assert is valid", PrimaryILOffset = 20, MethodILOffset = 0)]
    public void Loop3()
    {
      int i = 100;
      while (i > 0)
        i--;

      Contract.Assert(i <= 0);
    }

    [ClousotRegressionTest]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = @"assert is valid", PrimaryILOffset = 15, MethodILOffset = 0)]
    public void Loop4()
    {
      int i = 100;
      do
      {
        i--;
      } while (i > 0);

      // Prove it despite of the expression refinement in the guard
      Contract.Assert(i == 0);
    }

    [ClousotRegressionTest]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = @"assert is valid", PrimaryILOffset = 17, MethodILOffset = 0)]
    public void Loop5()
    {
      var x = 10;

      while (x >= 0)
      {
        x = x - 1;
      }
    
      // Prove it thanks to the threshold -1 hardcoded in Clousot
      Contract.Assert(x == -1);
    }

    [ClousotRegressionTest]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "assert is valid", PrimaryILOffset = 12, MethodILOffset = 0)]
    public void Loop6()
    {
      var x = 0;
      while (x != 103)
      {
        // Fixing a bug in the Threshold inference
        Contract.Assert(x < 1087);
        x++;
      }
    }
  }

  class TestNaN
  {

    #region Methods with Preconditions involving NaN

    // f == float.Nan is always false
    [ClousotRegressionTest]
#if CLOUSOT2
    [RegressionOutcome(Outcome=ProofOutcome.Top,Message="method Requires (including invariants) are unsatisfiable: TestIntervals.TestNaN.Pre_ImpossibleToSatisfy(System.Single)",PrimaryILOffset=13,MethodILOffset=0)]
#else
    [RegressionOutcome(Outcome=ProofOutcome.Top,Message=@"method Requires (including invariants) are unsatisfiable: TestIntervals.TestNaN.Pre_ImpossibleToSatisfy(System.Single)",PrimaryILOffset=14,MethodILOffset=0)]
#endif
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = @"The arguments of the comparison have a compatible precision", PrimaryILOffset = 6, MethodILOffset = 0)]
    public int Pre_ImpossibleToSatisfy(float f)
    {
      Contract.Requires(f == float.NaN);

      // F: We no not get any warning for the assert(false). is this a bug???
      // M: unless you use -show unreachable, you won't get an outcome here.

      Contract.Assert(false);

      return 3;
    }

    [ClousotRegressionTest]
#if CLOUSOT2
    [RegressionOutcome(Outcome=ProofOutcome.Top,Message="method Requires (including inherited requires and invariants) are unsatisfiable: TestIntervals.TestNaN.Pre_VirtualImpossibleToSatisfy(System.Single)",PrimaryILOffset=13,MethodILOffset=0)]
#else
    [RegressionOutcome(Outcome=ProofOutcome.Top,Message=@"method Requires (including inherited requires and invariants) are unsatisfiable: TestIntervals.TestNaN.Pre_VirtualImpossibleToSatisfy(System.Single)",PrimaryILOffset=14,MethodILOffset=0)]
#endif    
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = @"The arguments of the comparison have a compatible precision", PrimaryILOffset = 6, MethodILOffset = 0)]
    public virtual int Pre_VirtualImpossibleToSatisfy(float f)
    {
      Contract.Requires(f == float.NaN);

      // F: We no not get any warning for the assert(false). is this a bug???
      // M: unless you use -show unreachable, you won't get an outcome here.

      Contract.Assert(false);

      return 3;
    }

    // The precondition holdfs if f.IsNaN
    [ClousotRegressionTest]
    public int Pre_CanSatisfy(float f)
    {
      Contract.Requires(f.Equals(float.NaN));

      return 5;
    }

    // This always hold
    [ClousotRegressionTest]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = @"The arguments of the comparison have a compatible precision", PrimaryILOffset = 10, MethodILOffset = 0)]
    public int Pre_AlwaysSatisfied(double d)
    {
      Contract.Requires(d != double.NaN);

      return 4;
    }

    #endregion

    #region Callers

    [ClousotRegressionTest]
	#if CLOUSOT2 && NETFRAMEWORK_4_0
		[RegressionOutcome(Outcome=ProofOutcome.False,Message="requires is false: f == float.NaN",PrimaryILOffset=8,MethodILOffset=6)]
	#else
		[RegressionOutcome(Outcome = ProofOutcome.False, Message = @"requires is false: f == float.NaN", PrimaryILOffset = 8, MethodILOffset = 6)]
	#endif
    public void Call_Pre_ImpossibleToSatisfy()
    {
      int i = Pre_ImpossibleToSatisfy(float.NaN);
    }

    [ClousotRegressionTest]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = @"requires is valid", PrimaryILOffset = 12, MethodILOffset = 6)]
    public void Call_Pre_CanSatisfy()
    {
      int i = Pre_CanSatisfy(float.NaN);
    }

    [ClousotRegressionTest]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = @"requires is valid", PrimaryILOffset = 15, MethodILOffset = 10)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = @"requires is valid", PrimaryILOffset = 15, MethodILOffset = 26)]
    public void Call_Pre_AlwaysSatisfied()
    {
      int i = Pre_AlwaysSatisfied(double.NaN);
      int k = Pre_AlwaysSatisfied(2.34567);
    }
    #endregion
  }

  class UInts
  {
    [ClousotRegressionTest]
    [RegressionOutcome(Outcome = ProofOutcome.Top, Message = "assert unproven. The static checker determined that the condition 'input == 0' should hold on entry. Nevertheless, the condition may be too strong for the callers. If you think it is ok, add a precondition to document it: Contract.Requires(input == 0);", PrimaryILOffset = (int)22, MethodILOffset = 0)]
    static void Loop_int(int input)
    {
      int i = input;
      int count = 0;
      while (i > 0)
      {
        count++;
        i--;
      }

      Contract.Assert(input == 0);    // Should fail, as input may be negative
    }

    // Cannot prove it, as bgt_un is not recognized by Clousot at the moment. Should be fixed
    //    [ClousotRegressionTest]
    static void Loop_uint(uint input)
    {
      uint i = input;
      uint count = 0;
      while (i > 0)
      {
        count++;
        i--;
      }

      Contract.Assert(i == 0);  // That's true as everything is unsigned int, but Clousot cannot prove it yet
    }

    [ClousotRegressionTest]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = @"requires is valid", PrimaryILOffset = 11, MethodILOffset = 1)]
    static void Main()
    {
      Test(1); // <-- requires is true
    }

    static void Test(uint value)
    {
      Contract.Requires(value <= 0x80000000U);
    }
  }

  class Long
  {
    class MaxValueLong
    {
      [ClousotRegressionTest]
      [RegressionOutcome(Outcome = ProofOutcome.True, Message = @"assert is valid", PrimaryILOffset = 23, MethodILOffset = 0)]
      [RegressionOutcome(Outcome = ProofOutcome.True, Message = @"assert is valid", PrimaryILOffset = 37, MethodILOffset = 0)]
      public static void M(long l)
      {
        Contract.Requires(l == long.MaxValue);

        if (l == 2)
        {
          // Should be unreached, but we report to be valid, as it is an "Assert false"
          Contract.Assert(false);
        }
        else
        { // Should be ok
          Contract.Assert(l >= 0);
        }
      }
	  
	  [ClousotRegressionTest]
	  [RegressionOutcome(Outcome=ProofOutcome.False,Message="ensures is false: Contract.Result<object>() != null",PrimaryILOffset=11,MethodILOffset=36)]
      [RegressionOutcome(Outcome=ProofOutcome.Top,Message="ensures unproven: Contract.Result<object>() != null",PrimaryILOffset=11,MethodILOffset=42)]
	  public object Repro(object Value)
      {
        Contract.Ensures(Contract.Result<object>() != null);

        var l = (long)Value;
        if (l == long.MinValue) // We had an internal overflow in Clousot, which let it think l != MinValue all the times, for all longs...
        {
          return null;
        }
        return "f";
	  }
    }
  }

  class BasicRanges
  {
    public static void RequiresByte(byte b)
    {
      Contract.Requires(b >= Byte.MinValue);
      Contract.Requires(b <= Byte.MaxValue);
    }

    [ClousotRegressionTest]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = @"requires is valid", PrimaryILOffset = 7, MethodILOffset = 1)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = @"requires is valid", PrimaryILOffset = 23, MethodILOffset = 1)]
    public static void CallRequiresByte(byte b)
    {
      RequiresByte(b);
    }
  }

  class BitwiseOperators
  {
    [ClousotRegressionTest]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = @"assert is valid", PrimaryILOffset = 41, MethodILOffset = 0)]
    public void AlexeyR(int shift)
    {
      Contract.Requires(shift >= 0);
      Contract.Requires(shift <= 2);

      int x = int.MaxValue >> shift * 8;  // C# generarates ((shift * 8) & 31)
      
      Contract.Assert(x > 0); 
    }
  }
}

namespace FromMscorlib
{


  // This test exercises the "big" numbers
  public class DateTime
  {
    private static int[] DaysToMonth365, DaysToMonth366;

    static DateTime()
    {
      DaysToMonth365 = new int[] { 0, 0x1f, 0x3b, 90, 120, 0x97, 0xb5, 0xd4, 0xf3, 0x111, 0x130, 0x14e, 0x16d };
      DaysToMonth366 = new int[] { 0, 0x1f, 60, 0x5b, 0x79, 0x98, 0xb6, 0xd5, 0xf4, 0x112, 0x131, 0x14f, 0x16e };
    }

    private ulong dateData;

    public DateTime(ulong dateData)
    {
      Contract.Requires(dateData >= 0);
      this.dateData = dateData;
    }

    private long InternalTicks
    {
      // ok
      [ClousotRegressionTest]
      [RegressionOutcome(Outcome=ProofOutcome.True,Message="ensures is valid",PrimaryILOffset=12,MethodILOffset=33)]
      get
      {
        Contract.Ensures(Contract.Result<long>() >= 0);
        return (((long)this.dateData) & 0x3fffffffffffffffL);
      }
    }

    // All the division by zero should be ok
    // 3 (out of 4) array accesses should raise a warning
    // One assertion fails because of lack of object invariant
    [RegressionOutcome(Outcome=ProofOutcome.Top,Message="Array access might be below the lower bound",PrimaryILOffset=204,MethodILOffset=0)]
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
    [ClousotRegressionTest]    
    private int GetDatePart(int part)
    {
      Contract.Assert(0xc92a69c000L != 0);    // This is converted by the compiler to "1", and it undercovered a bug in the CheckIfHolds of intervals, so we keep it

      int num2 = (int)(this.InternalTicks / 0xc92a69c000L);
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
      Contract.Assert(numArray.Length == 13);

      int index = num2 >> 6;
      
      // Here should prove index < 13, which is quite hard
      while (num2 >= numArray[index])
      {
        index++;
      }
      if (part == 2)
      {
        return index;
      }
      return ((num2 - numArray[index - 1]) + 1);
    } 
  }

  public class Enum
  {
    [ClousotRegressionTest]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "Array creation : ok", PrimaryILOffset = 3, MethodILOffset = 0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"No overflow (caused by a negative array size)",PrimaryILOffset=3,MethodILOffset=0)]
#if NETFRAMEWORK_4_0
    [RegressionOutcome(Outcome=ProofOutcome.True,Message="Lower bound access ok",PrimaryILOffset=28,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.Top,Message="Array access might be above the upper bound",PrimaryILOffset=28,MethodILOffset=0)]
#else
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "Lower bound access ok", PrimaryILOffset = 32, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.Top, Message = "Array access might be above the upper bound", PrimaryILOffset = 32, MethodILOffset = 0)]
#endif
    private static object GetHashEntry(Type enumType, bool[] flds, ulong val)
    {
      ulong[] values = new ulong[flds.Length];
      string[] names = new string[flds.Length]; // /optimize+ removes this array creation,

      for (int i = 1; i < values.Length; i++)
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

}


namespace Repro
{
  class AlexeyR
  {
    [ClousotRegressionTest]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"Array creation : ok",PrimaryILOffset=1,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"Lower bound access ok",PrimaryILOffset=9,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"Upper bound access ok",PrimaryILOffset=9,MethodILOffset=0)]	
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"assert is valid",PrimaryILOffset=22,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"No overflow (caused by a negative array size)",PrimaryILOffset=1,MethodILOffset=0)]
    public void ldelem()
    {
      byte[] buf = new byte[1];
      byte val = buf[0];
      Contract.Assert(val <= byte.MaxValue); 
    }
  }
  
}

