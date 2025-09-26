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
import org.checkerframework.checker.index.qual.IndexOrLow;
import org.checkerframework.checker.index.qual.LTLengthOf;
import org.checkerframework.checker.index.qual.NonNegative;
import org.checkerframework.common.value.qual.IntRange;
import org.checkerframework.common.value.qual.IntVal;
import org.checkerframework.dataflow.qual.Pure;
class WebExamples_Tutorial
  {

    /*[ClousotRegressionTest]   
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"valid non-null reference (as array)",PrimaryILOffset=126,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"valid non-null reference (as array)",PrimaryILOffset=114,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"valid non-null reference (as array)",PrimaryILOffset=46,MethodILOffset=131)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"valid non-null reference (as array)",PrimaryILOffset=46,MethodILOffset=119)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"valid non-null reference (as array)",PrimaryILOffset=96,MethodILOffset=119)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"ensures is valid",PrimaryILOffset=35,MethodILOffset=131)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"ensures is valid",PrimaryILOffset=50,MethodILOffset=131)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"ensures is valid",PrimaryILOffset=77,MethodILOffset=131)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"ensures is valid",PrimaryILOffset=35,MethodILOffset=119)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"ensures is valid",PrimaryILOffset=50,MethodILOffset=119)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"ensures is valid",PrimaryILOffset=77,MethodILOffset=119)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"ensures is valid",PrimaryILOffset=103,MethodILOffset=119)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"ensures is valid",PrimaryILOffset=103,MethodILOffset=131)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"Lower bound access ok",PrimaryILOffset=114,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"Upper bound access ok",PrimaryILOffset=114,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"Lower bound access ok",PrimaryILOffset=96,MethodILOffset=119)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"Upper bound access ok",PrimaryILOffset=96,MethodILOffset=119)]
    [Pure]*/
	@Pure
    public static @IndexOrLow("#1") int LinearSearch(int[] array, @NonNegative int from, int value)
    {
      //N: Contract.Requires(array != null);
      //Y: Contract.Requires(from >= 0);

      //Y: Contract.Ensures(Contract.Result<int>() >= -1);
      //Y: Contract.Ensures(Contract.Result<int>() < array.Length);

      //N: Contract.Ensures(Contract.Result<int>() == -1 || Contract.Result<int>() >= from); // We need it!
      //N: Contract.Ensures(Contract.Result<int>() == -1 || array[Contract.Result<int>()] == value);

      for (int i = from; i < array.length; i++)
      {
        if (array[i] == value)
          return i;
      }
      return -1;
    }


    /*[ClousotRegressionTest]   
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"valid non-null reference (as field receiver)",PrimaryILOffset=8,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"valid non-null reference (as field receiver)",PrimaryILOffset=14,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"valid non-null reference (as field receiver)",PrimaryILOffset=8,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"valid non-null reference (as field receiver)",PrimaryILOffset=127,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"valid non-null reference (as array)",PrimaryILOffset=132,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"valid non-null reference (as field receiver)",PrimaryILOffset=63,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"valid non-null reference (as field receiver)",PrimaryILOffset=8,MethodILOffset=137)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"valid non-null reference (as field receiver)",PrimaryILOffset=93,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"valid non-null reference (as field receiver)",PrimaryILOffset=100,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"valid non-null reference (as array)",PrimaryILOffset=106,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"valid non-null reference (as array)",PrimaryILOffset=107,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"valid non-null reference (as field receiver)",PrimaryILOffset=109,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"valid non-null reference (as array)",PrimaryILOffset=116,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"valid non-null reference (as field receiver)",PrimaryILOffset=8,MethodILOffset=120)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"requires is valid",PrimaryILOffset=7,MethodILOffset=70)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"requires is valid",PrimaryILOffset=19,MethodILOffset=70)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"ensures is valid",PrimaryILOffset=53,MethodILOffset=137)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"assert is valid",PrimaryILOffset=87,MethodILOffset=0)] 
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"ensures is valid",PrimaryILOffset=53,MethodILOffset=120)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"Lower bound access ok",PrimaryILOffset=106,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"Upper bound access ok",PrimaryILOffset=106,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"Lower bound access ok",PrimaryILOffset=107,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"Upper bound access ok",PrimaryILOffset=107,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"Lower bound access ok",PrimaryILOffset=116,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"Upper bound access ok",PrimaryILOffset=116,MethodILOffset=0)]*/
   public static int ZerosAtBeginning(int[] array)
    {
      //N: Contract.Requires(array != null);
      //N: Contract.Ensures(Contract.ForAll(0, Contract.Result<int>(), index => array[index] == 0));

      int i = 0;
      while (i < array.length)
      {
        int nextZero = LinearSearch(array, i, 0);
        if (nextZero != -1)
        {
          //N: Contract.Assert(nextZero >= i);
          array[nextZero] = array[i];
          array[i] = 0;
        }
        else
        {
          return i;
        }

        i++;
      }

      return i;
    }

  }

  class  WebExamples_ArrayBounds
  {
    // What can possibly go wrong here?
    /*[ClousotRegressionTest]
    [RegressionOutcome(Outcome=ProofOutcome.Top,Message=@"Possible use of a null array 'a'. The static checker determined that the condition 'a != null' should hold on entry. Nevertheless, the condition may be too strong for the callers. If you think it is ok, add a precondition to document it: Contract.Requires(a != null);",PrimaryILOffset=9,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.Top,Message=@"Array access might be below the lower bound",PrimaryILOffset=9,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.Top,Message=@"Array access might be above the upper bound",PrimaryILOffset=9,MethodILOffset=0)]*/
    static int Sum(int[] a, int start, int numItems)
    {
      int result = 0;
      for (int i = start; i < start + numItems; i++)
    	// :: error: (array.access.unsafe.high.range) :: error: (array.access.unsafe.low)
	result += a[i];
      return result;
    }

    /*[ClousotRegressionTest]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"valid non-null reference (as array)",PrimaryILOffset=40,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"valid non-null reference (as array)",PrimaryILOffset=61,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"Lower bound access ok",PrimaryILOffset=61,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"Upper bound access ok",PrimaryILOffset=61,MethodILOffset=0)]*/
    static int SumFixed(int[] a, @NonNegative @LTLengthOf(value="#1", offset="#3-1")int start, @NonNegative int numItems)
    {
      //N: Contract.Requires(a != null);
      //Y: Contract.Requires(0 <= start);
    //Y: Contract.Requires(0 <= numItems);
    	//Y: Contract.Requires(start + numItems <= a.Length);
      
      int result = 0;
      for (int i = start; i < start + numItems; i++)
	result += a[i];
      return result;
    }
  }


  class MinIndexExample
  {
    // Compute the index of the minimal element in data, or return -1
    /*[ClousotRegressionTest]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"valid non-null reference (as array)",PrimaryILOffset=73,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"valid non-null reference (as array)",PrimaryILOffset=59,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"valid non-null reference (as array)",PrimaryILOffset=62,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"valid non-null reference (as array)",PrimaryILOffset=34,MethodILOffset=78)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"ensures is valid",PrimaryILOffset=23,MethodILOffset=78)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"ensures is valid",PrimaryILOffset=38,MethodILOffset=78)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"Lower bound access ok",PrimaryILOffset=59,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"Upper bound access ok",PrimaryILOffset=59,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"Lower bound access ok",PrimaryILOffset=62,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"Upper bound access ok",PrimaryILOffset=62,MethodILOffset=0)]*/
    static @IndexOrLow("#1") int MinIndex(int[] data)
    {
      //N: Contract.Requires(data != null);
      //Y: Contract.Ensures(Contract.Result<int>() >= -1);
      //Y: Contract.Ensures(Contract.Result<int>() < data.Length);
      
    	@IndexOrLow("#1") int result = -1;
      for (int i = 0; i < data.length; i++)
        {
	  if (result < 0) { result = i; }
	  else if (data[i] < data[result]) { result = i; }
        }
      return result;
    }

    /*[ClousotRegressionTest]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"valid non-null reference (as array)",PrimaryILOffset=21,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"requires is valid",PrimaryILOffset=7,MethodILOffset=13)]
    [RegressionOutcome(Outcome=ProofOutcome.Top,Message=@"Array access might be below the lower bound",PrimaryILOffset=21,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"Upper bound access ok",PrimaryILOffset=21,MethodILOffset=0)]*/
    static int BadUser(int[] data)
    {
      //N: Contract.Requires(data != null);
      
      int minIndex = MinIndex(data);
   // :: error: (array.access.unsafe.low)
      int minValue = data[minIndex];
      return minValue;
    }

    /*[ClousotRegressionTest]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"valid non-null reference (as array)",PrimaryILOffset=25,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"requires is valid",PrimaryILOffset=7,MethodILOffset=13)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"Lower bound access ok",PrimaryILOffset=25,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"Upper bound access ok",PrimaryILOffset=25,MethodILOffset=0)]*/
    static int GoodUser(int[] data)
    {
      //N: Contract.Requires(data != null);
      
    	int minIndex = MinIndex(data);
      if (minIndex >= 0)
        {
    	  int minValue = data[minIndex];
	  return minValue;
        }
      return Integer.MIN_VALUE;
    }
  }

  class Search
  {

    /*[ClousotRegressionTest]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"Lower bound access ok",PrimaryILOffset=28,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.Top,Message=@"Array access might be above the upper bound",PrimaryILOffset=28,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"Lower bound access ok",PrimaryILOffset=36,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"Upper bound access ok",PrimaryILOffset=36,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"valid non-null reference (as array)",PrimaryILOffset=15,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"valid non-null reference (as array)",PrimaryILOffset=28,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"valid non-null reference (as array)",PrimaryILOffset=36,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"No underflow",PrimaryILOffset=22,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.Top,Message=@"Possible overflow in the arithmetic operation",PrimaryILOffset=22,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"Division by zero ok",PrimaryILOffset=24,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"No overflow",PrimaryILOffset=24,MethodILOffset=0)]*/
    public static int BinarySearch_Wrong(int[] arr, int what)
    {
      //N: Contract.Requires(arr != null);

    	int inf = 0;
    	int sup = arr.length;
      
      while (inf <= sup)
      {
    	  int mid = /*checked*/((inf + sup)/ 2);
    	// :: error: (array.access.unsafe.high.range)
        if (arr[mid] == what)
          return what;
// :: error: (array.access.unsafe.high.range)
        if (arr[mid] < what)
        {
          sup = mid - 1;
        }
        else
        {
          inf = mid+1;
        }
      }
      return -1;
    }

    /*[ClousotRegressionTest]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"Lower bound access ok",PrimaryILOffset=32,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"Upper bound access ok",PrimaryILOffset=32,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"Lower bound access ok",PrimaryILOffset=40,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"Upper bound access ok",PrimaryILOffset=40,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"valid non-null reference (as array)",PrimaryILOffset=15,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"valid non-null reference (as array)",PrimaryILOffset=32,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"valid non-null reference (as array)",PrimaryILOffset=40,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"No underflow",PrimaryILOffset=25,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"No overflow",PrimaryILOffset=25,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"Division by zero ok",PrimaryILOffset=27,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"No overflow",PrimaryILOffset=27,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"No underflow",PrimaryILOffset=28,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"No overflow",PrimaryILOffset=28,MethodILOffset=0)]*/
    public static int BinarySearch_Correct(int[] arr, int what)
    {
      //N: Contract.Requires(arr != null);

    	int inf = 0;
    	int sup = arr.length - 1;
      
      while (inf <= sup)
      {
    	  int len = sup - inf;
    	  @IndexFor("arr") int mid = /*checked*/(inf + (sup - inf) / 2);
        if (arr[mid] == what)
          return what;
        if (arr[mid] < what)
        {
          sup = mid - 1;
        }
        else
        {
          inf = mid + 1;
        }
      }
      return -1;
    }
  }


 class VMCAI_VMCAI
  {
    /*[ClousotRegressionTest]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"Array creation : ok",PrimaryILOffset=178,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"Lower bound access ok",PrimaryILOffset=192,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"Upper bound access ok",PrimaryILOffset=192,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"Lower bound access ok",PrimaryILOffset=217,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"Upper bound access ok",PrimaryILOffset=217,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"Lower bound access ok",PrimaryILOffset=247,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"Upper bound access ok",PrimaryILOffset=247,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"Array creation : ok",PrimaryILOffset=259,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"Lower bound access ok",PrimaryILOffset=278,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"Upper bound access ok",PrimaryILOffset=278,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"Lower bound access ok",PrimaryILOffset=279,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"Upper bound access ok",PrimaryILOffset=279,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"valid non-null reference (as array)",PrimaryILOffset=176,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"valid non-null reference (as array)",PrimaryILOffset=254,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"valid non-null reference (as array)",PrimaryILOffset=192,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"valid non-null reference (as array)",PrimaryILOffset=217,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"valid non-null reference (as array)",PrimaryILOffset=247,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"valid non-null reference (as array)",PrimaryILOffset=278,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"valid non-null reference (as array)",PrimaryILOffset=279,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"valid non-null reference (as array)",PrimaryILOffset=44,MethodILOffset=293)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"valid non-null reference (as array)",PrimaryILOffset=66,MethodILOffset=293)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"No overflow (caused by a negative array size)",PrimaryILOffset=178,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"No overflow (caused by a negative array size)",PrimaryILOffset=259,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"ensures is valid",PrimaryILOffset=20,MethodILOffset=293)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"ensures is valid",PrimaryILOffset=33,MethodILOffset=293)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"ensures is valid",PrimaryILOffset=51,MethodILOffset=293)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"ensures is valid",PrimaryILOffset=70,MethodILOffset=293)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"ensures is valid",PrimaryILOffset=115,MethodILOffset=293)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"ensures is valid",PrimaryILOffset=160,MethodILOffset=293)]*/
    public @IntRange(from = 'a', to='z') char[] Sanitize(char[] str, /*ref*/ int lower, /*ref*/ int upper)
    {
    	//N: Contract.Requires(str != null);

      //Y: Contract.Ensures(upper >= 0);
      //Y: Contract.Ensures(lower >= 0);
      //Y: Contract.Ensures(lower + upper <= str.Length);
      //N: Contract.Ensures(lower + upper == Contract.Result<char[]>().Length);
      //Y: Contract.Ensures(Contract.ForAll(0, lower+upper, index => 'a' <= Contract.Result<char[]>()[index]));
      //Y: Contract.Ensures(Contract.ForAll(0, lower + upper, index => Contract.Result<char[]>()[index] <= 'z'));

      upper = lower = 0;

      char[] tmp = new char[str.length];

      int j = 0;
      for (int i = 0; i < str.length; i++)
      {
    	  char ch = str[i];

        if ('a' <= ch && ch <= 'z') { lower++; tmp[j++] = ch;}
        else if ('A' <= ch && ch<= 'Z') { upper++; tmp[j++] = (char)(ch | ' ');}
      }

      char[]  result = new char[j];
      for (int i = 0; i < j; i++)
      {
        result[i] = tmp[i];
      }
      
      /*Ensures*/
      @NonNegative int _1 = upper;
      @NonNegative int _2 = lower;
      @LTLengthOf(value="result", offset="upper-1") int _3 = lower;
      /*End ensures*/
      
      
      return result;
    }
 
}


  class Bagnara_LeviTalk
  {
    /*[ClousotRegressionTest]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message="Lower bound access ok",PrimaryILOffset=51,MethodILOffset=0)]
[RegressionOutcome(Outcome=ProofOutcome.True,Message="Upper bound access ok",PrimaryILOffset=51,MethodILOffset=0)]
[RegressionOutcome(Outcome=ProofOutcome.True,Message="Lower bound access ok",PrimaryILOffset=77,MethodILOffset=0)]
[RegressionOutcome(Outcome=ProofOutcome.True,Message="Upper bound access ok",PrimaryILOffset=77,MethodILOffset=0)]
[RegressionOutcome(Outcome=ProofOutcome.True,Message="Lower bound access ok",PrimaryILOffset=63,MethodILOffset=0)]
[RegressionOutcome(Outcome=ProofOutcome.True,Message="Upper bound access ok",PrimaryILOffset=63,MethodILOffset=0)]
[RegressionOutcome(Outcome=ProofOutcome.True,Message="Lower bound access ok",PrimaryILOffset=64,MethodILOffset=0)]
[RegressionOutcome(Outcome=ProofOutcome.True,Message="Upper bound access ok",PrimaryILOffset=64,MethodILOffset=0)]
[RegressionOutcome(Outcome=ProofOutcome.True,Message="Lower bound access ok",PrimaryILOffset=84,MethodILOffset=0)]
[RegressionOutcome(Outcome=ProofOutcome.True,Message="Upper bound access ok",PrimaryILOffset=84,MethodILOffset=0)]
[RegressionOutcome(Outcome=ProofOutcome.True,Message="valid non-null reference (as array)",PrimaryILOffset=14,MethodILOffset=0)]
[RegressionOutcome(Outcome=ProofOutcome.True,Message="valid non-null reference (as array)",PrimaryILOffset=51,MethodILOffset=0)]
[RegressionOutcome(Outcome=ProofOutcome.True,Message="valid non-null reference (as array)",PrimaryILOffset=77,MethodILOffset=0)]
[RegressionOutcome(Outcome=ProofOutcome.True,Message="valid non-null reference (as array)",PrimaryILOffset=63,MethodILOffset=0)]
[RegressionOutcome(Outcome=ProofOutcome.True,Message="valid non-null reference (as array)",PrimaryILOffset=64,MethodILOffset=0)]
[RegressionOutcome(Outcome=ProofOutcome.True,Message="valid non-null reference (as array)",PrimaryILOffset=84,MethodILOffset=0)]
[RegressionOutcome(Outcome=ProofOutcome.True,Message="Division by zero ok",PrimaryILOffset=95,MethodILOffset=0)]
[RegressionOutcome(Outcome=ProofOutcome.True,Message="No overflow",PrimaryILOffset=95,MethodILOffset=0)]*/
    public void ShellSort(@IndexOrHigh("#2") int n, int[] array)
    {
      //N: Contract.Requires(array != null);
      //YN: Contract.Requires(n == array.Length); // just to adapt Bagnara example faithfully

      int i, j; int h, B; //TODO: revert

      h = 1;
      while (h * 3 + 1 < n)
      {
        h = 3 * h + 1;
      }
      while (h > 0)
      {
        i = h - 1;
        while(i < n)
        {
          B = array[i];
          j = i;
          while (j >= h && array[j - h] > B)
          {
            array[j] = array[j - h];
            j = j - h;
          }
          array[j] = B;
          i++;
        }
        h = h / 3;
      }
    }
  }  
