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

import org.checkerframework.common.value.qual.MinLen;

class ExamplesPurtiy_Simple {
	
	  /*[ClousotRegressionTest]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"Lower bound access ok",PrimaryILOffset=27,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"Upper bound access ok",PrimaryILOffset=27,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"valid non-null reference (as array)",PrimaryILOffset=13,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"valid non-null reference (as array)",PrimaryILOffset=27,MethodILOffset=0)]*/
    public void ChangeFirstElementNoPure( int @MinLen(1)[] arr,  Object[] untouched)
    {
      //N: Contract.Requires(arr != null);
      //Y: Contract.Requires(arr.Length > 0);

      arr[0] = 123;
    }
   /* [ClousotRegressionTest]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"Lower bound access ok",PrimaryILOffset=27,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"Upper bound access ok",PrimaryILOffset=27,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"valid non-null reference (as array)",PrimaryILOffset=13,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"valid non-null reference (as array)",PrimaryILOffset=27,MethodILOffset=0)]*/
    // Nothing to suggest
	public void ChangeFirstElement(int @MinLen(1) [] arr, Object[] untouched) {
	      //N: Contract.Requires(arr != null);
	      //Y: Contract.Requires(arr.Length > 0);
		arr[0] = 123;
	}

    /*[ClousotRegressionTest]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"Lower bound access ok",PrimaryILOffset=23,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"Upper bound access ok",PrimaryILOffset=23,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"valid non-null reference (as array)",PrimaryILOffset=30,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"valid non-null reference (as array)",PrimaryILOffset=23,MethodILOffset=0)]*/
    // Nothing to suggest
	public void ChangeAllTheElements(int[] arr) {

		 //N: Contract.Requires(arr != null);
		for (int i = 0; i < arr.length; i++) {
			arr[i] = 1234;
		}
	}

    /*[ClousotRegressionTest]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"Lower bound access ok",PrimaryILOffset=9,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"Upper bound access ok",PrimaryILOffset=9,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.Top,Message=@"Possible use of a null array 'arr'",PrimaryILOffset=18,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"valid non-null reference (as array)",PrimaryILOffset=9,MethodILOffset=0)]
    [RegressionOutcome("Contract.Requires(arr != null);")]
    [RegressionOutcome("Consider adding the [Pure] attribute to the parameter arr")]*/
	public int Unchanged(int[] arr) {
		int sum = 0;
		for (int i = 0; i < arr.length; i++) {
			sum += arr[i];
		}

		return sum;
	}

}

class ForAllPreconditionInference_Tests {
	/*[ClousotRegressionTest]
			    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"Lower bound access ok",PrimaryILOffset=6,MethodILOffset=0)]
			    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"Upper bound access ok",PrimaryILOffset=6,MethodILOffset=0)]
			    [RegressionOutcome(Outcome=ProofOutcome.Top,Message=@"Possible use of a null array 's'",PrimaryILOffset=24,MethodILOffset=0)]
			    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"valid non-null reference (as array)",PrimaryILOffset=6,MethodILOffset=0)]
			    [RegressionOutcome(Outcome=ProofOutcome.Top,Message=@"assert unproven",PrimaryILOffset=13,MethodILOffset=0)]
			    [RegressionOutcome("Consider adding the [Pure] attribute to the parameter s")]
			    [RegressionOutcome("Contract.Requires(s != null);")]
			    [RegressionOutcome("Contract.Requires(Contract.Forall(s, 0, s.Length, s[i] != null));")]*/
			    public void TestAllElements(String[] s)
			    {
			      for (int i = 0; i < s.length; i++)
			      {
			    	//N: Contract.Assert(s[i] != null);
			      }
			    }

			    /*[ClousotRegressionTest]
			    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"Lower bound access ok",PrimaryILOffset=6,MethodILOffset=0)]
			    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"Upper bound access ok",PrimaryILOffset=6,MethodILOffset=0)]
			    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"Lower bound access ok",PrimaryILOffset=21,MethodILOffset=0)]
			    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"Upper bound access ok",PrimaryILOffset=21,MethodILOffset=0)]
			    [RegressionOutcome(Outcome=ProofOutcome.Top,Message=@"Possible use of a null array 's'",PrimaryILOffset=28,MethodILOffset=0)]
			    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"valid non-null reference (as array)",PrimaryILOffset=6,MethodILOffset=0)]
			    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"valid non-null reference (as array)",PrimaryILOffset=21,MethodILOffset=0)]
			    [RegressionOutcome(Outcome=ProofOutcome.Top,Message=@"assert unproven",PrimaryILOffset=13,MethodILOffset=0)]
			    [RegressionOutcome("Contract.Requires(s != null);")]
			    [RegressionOutcome("Contract.Requires(Contract.Forall(s, 0, s.Length, s[i] != null));")]*/
			    public void TestAllElementsAndChangeThem(String[] s)
			    {
			      for (int i = 0; i < s.length; i++)
			      {
			    	//N: Contract.Assert(s[i] != null);
			        s[i] = null;
			      }
			    }

			    /*[ClousotRegressionTest]
			    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"Lower bound access ok",PrimaryILOffset=6,MethodILOffset=0)]
			    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"Upper bound access ok",PrimaryILOffset=6,MethodILOffset=0)]
			    [RegressionOutcome(Outcome=ProofOutcome.Top,Message=@"Possible use of a null array 's'",PrimaryILOffset=24,MethodILOffset=0)]
			    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"valid non-null reference (as array)",PrimaryILOffset=6,MethodILOffset=0)]
			    [RegressionOutcome("Contract.Requires(s != null);")]
			    [RegressionOutcome("Consider adding the [Pure] attribute to the parameter s")]
			    [RegressionOutcome("Contract.Requires(Contract.Forall(s, 0, s.Length, s[i] != null));")]*/
			    public void AssumeAllElements(String[] s)
			    {
			      for (int i = 0; i < s.length; i++)
			      {
			        //N: Contract.Assume(s[i] != null);
			      }
			    }

			    /*[ClousotRegressionTest]
			    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"Lower bound access ok",PrimaryILOffset=6,MethodILOffset=0)]
			    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"Upper bound access ok",PrimaryILOffset=6,MethodILOffset=0)]
			    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"Lower bound access ok",PrimaryILOffset=21,MethodILOffset=0)]
			    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"Upper bound access ok",PrimaryILOffset=21,MethodILOffset=0)]
			    [RegressionOutcome(Outcome=ProofOutcome.Top,Message=@"Possible use of a null array 's'",PrimaryILOffset=28,MethodILOffset=0)]
			    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"valid non-null reference (as array)",PrimaryILOffset=6,MethodILOffset=0)]
			    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"valid non-null reference (as array)",PrimaryILOffset=21,MethodILOffset=0)]
			    [RegressionOutcome("Contract.Requires(s != null);")]
			    [RegressionOutcome("Contract.Requires(Contract.Forall(s, 0, s.Length, s[i] != null));")]*/
			    public void AssumeAllElementsAndChangeThem(String[] s)
			    {
			      for (int i = 0; i < s.length; i++)
			      {
			    	//N: Contract.Assume(s[i] != null);
			        s[i] = null;
			      }
			    }

			    /*[ClousotRegressionTest]
			    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"Lower bound access ok",PrimaryILOffset=2,MethodILOffset=0)]
			    [RegressionOutcome(Outcome=ProofOutcome.Top,Message=@"Array access might be above the upper bound",PrimaryILOffset=2,MethodILOffset=0)]
			    [RegressionOutcome(Outcome=ProofOutcome.Top,Message=@"Possible use of a null array 's'",PrimaryILOffset=2,MethodILOffset=0)]
			    [RegressionOutcome(Outcome=ProofOutcome.Top,Message=@"assert unproven",PrimaryILOffset=9,MethodILOffset=0)]
			    [RegressionOutcome("Contract.Requires(3 < s.Length);")]
			    [RegressionOutcome("Contract.Requires(s != null);")]
			    [RegressionOutcome("Consider adding the [Pure] attribute to the parameter s")]
			    [RegressionOutcome("Contract.Requires(Contract.Forall(s, 3, 4, s[i] != null));")]*/
			    public void AssertOneElement(String[] s)
			    {
			      //N: Contract.Assert(s[3] != null);
			    }

			    /*[ClousotRegressionTest]
			    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"Lower bound access ok",PrimaryILOffset=3,MethodILOffset=0)]
			    [RegressionOutcome(Outcome=ProofOutcome.Top,Message=@"Array access might be above the upper bound",PrimaryILOffset=3,MethodILOffset=0)]
			    [RegressionOutcome(Outcome=ProofOutcome.Top,Message=@"Possible use of a null array 's'",PrimaryILOffset=3,MethodILOffset=0)]
			    [RegressionOutcome("Contract.Requires(12 < s.Length);")]
			    [RegressionOutcome("Contract.Requires(s != null);")]
			    [RegressionOutcome("Consider adding the [Pure] attribute to the parameter s")]*/
			    public int DoNotTestOneElement(String[] s)
			    {
			      int v;
			      // :: error: (array.access.unsafe.high.constant)
			      if (s[12] == null)
			      {
			        v = 0;
			      }
			      else
			      {
			        v = 1;
			      }

			      return v;
			    }

			    /*[ClousotRegressionTest]
			    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"Lower bound access ok",PrimaryILOffset=9,MethodILOffset=0)]
			    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"Upper bound access ok",PrimaryILOffset=9,MethodILOffset=0)]
			    [RegressionOutcome(Outcome=ProofOutcome.Top,Message=@"Possible use of a null array 's'",PrimaryILOffset=23,MethodILOffset=0)]
			    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"valid non-null reference (as array)",PrimaryILOffset=9,MethodILOffset=0)]
			    [RegressionOutcome(Outcome=ProofOutcome.Top,Message=@"Possibly calling a method on a null reference",PrimaryILOffset=10,MethodILOffset=0)]
			    [RegressionOutcome("Contract.Requires(s != null);")]
			    [RegressionOutcome("Consider adding the [Pure] attribute to the parameter s")]
			    [RegressionOutcome("Contract.Requires(Contract.Forall(s, 0, s.Length, s[i] != null));")]*/
	public int Sum(String[] s) {
		int sum = 0;
		for (int i = 0; i < s.length; i++) {
			sum += s[i].length();
		}

		return sum;
	}

			    /*[ClousotRegressionTest]
			    	    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"Lower bound access ok",PrimaryILOffset=9,MethodILOffset=0)]
			    	    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"Upper bound access ok",PrimaryILOffset=9,MethodILOffset=0)]
			    	    [RegressionOutcome(Outcome=ProofOutcome.Top,Message=@"Possible use of a null array 's'",PrimaryILOffset=23,MethodILOffset=0)]
			    	    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"valid non-null reference (as array)",PrimaryILOffset=9,MethodILOffset=0)]
			    	    [RegressionOutcome(Outcome=ProofOutcome.Top,Message=@"Possibly accessing a field on a null reference",PrimaryILOffset=10,MethodILOffset=0)]
			    	    [RegressionOutcome("Contract.Requires(s != null);")]
			    	    [RegressionOutcome("Consider adding the [Pure] attribute to the parameter s")]
			    	    [RegressionOutcome("Contract.Requires(Contract.Forall(s, 0, s.Length, s[i] != null));")]*/
	public int Sum(Dummy[] s) {
		int sum = 0;
		for (int i = 0; i < s.length; i++) {
			sum += s[i].f;
		}

		return sum;
	}

			    /*[ClousotRegressionTest]
			    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"Lower bound access ok",PrimaryILOffset=9,MethodILOffset=0)]
			    [RegressionOutcome(Outcome=ProofOutcome.Top,Message=@"Array access might be above the upper bound",PrimaryILOffset=9,MethodILOffset=0)]
			    [RegressionOutcome(Outcome=ProofOutcome.Top,Message=@"Possible use of a null array 's'",PrimaryILOffset=9,MethodILOffset=0)]
			    [RegressionOutcome(Outcome=ProofOutcome.Top,Message=@"Possibly accessing a field on a null reference",PrimaryILOffset=10,MethodILOffset=0)]
			    [RegressionOutcome("Contract.Requires(s != null);")]
			    [RegressionOutcome("Consider adding the [Pure] attribute to the parameter s")]*/
	public int Sum(Dummy[] s, int end) {
		int sum = 0;
		for (int i = 0; i < end; i++) {
			// :: error: (array.access.unsafe.high.range)
			sum += s[i].f;
		}

		return sum;

	}
			    /*[ClousotRegressionTest]
			    	    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"Lower bound access ok",PrimaryILOffset=15,MethodILOffset=0)]
			    	    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"Upper bound access ok",PrimaryILOffset=15,MethodILOffset=0)]
			    	    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"valid non-null reference (as array)",PrimaryILOffset=30,MethodILOffset=0)]
			    	    [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"valid non-null reference (as array)",PrimaryILOffset=15,MethodILOffset=0)]
			    	    [RegressionOutcome("Consider adding the [Pure] attribute to the parameter genericArguments")]
			    	    [RegressionOutcome("Contract.Requires(Contract.Forall(genericArguments, 0, genericArguments.Length, genericArguments[i] != null));")]*/

	static void SanityCheckArguments(Object[] genericArguments) {
		if (genericArguments == null) {
			throw new IllegalArgumentException();
		}
		for (int i = 0; i < genericArguments.length; i++) {
			if (genericArguments[i] == null) {
				throw new IllegalArgumentException();
			}
		}
	}

	public class Dummy {
		public int f;
	}
}