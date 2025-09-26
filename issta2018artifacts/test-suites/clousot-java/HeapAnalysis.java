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

public class HeapAnalysis {
	  /*[ClousotRegressionTest]
			  [RegressionOutcome(Outcome = ProofOutcome.True, Message = "Lower bound access ok", PrimaryILOffset = 66, MethodILOffset = 0)]
			  [RegressionOutcome(Outcome = ProofOutcome.True, Message = "Upper bound access ok", PrimaryILOffset = 66, MethodILOffset = 0)]
			  [RegressionOutcome(Outcome = ProofOutcome.True, Message = "Lower bound access ok", PrimaryILOffset = 69, MethodILOffset = 0)]
			  [RegressionOutcome(Outcome = ProofOutcome.True, Message = "Upper bound access ok", PrimaryILOffset = 69, MethodILOffset = 0)]
			  [RegressionOutcome(Outcome = ProofOutcome.True, Message = "Lower bound access ok", PrimaryILOffset = 83, MethodILOffset = 0)]
			  [RegressionOutcome(Outcome = ProofOutcome.True, Message = "Upper bound access ok", PrimaryILOffset = 83, MethodILOffset = 0)]
			  [RegressionOutcome(Outcome = ProofOutcome.True, Message = "Lower bound access ok", PrimaryILOffset = 87, MethodILOffset = 0)]
			  [RegressionOutcome(Outcome = ProofOutcome.True, Message = "Upper bound access ok", PrimaryILOffset = 87, MethodILOffset = 0)]
			  [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as array)", PrimaryILOffset = 29, MethodILOffset = 0)]
			  [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as array)", PrimaryILOffset = 54, MethodILOffset = 0)]
			  [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as array)", PrimaryILOffset = 66, MethodILOffset = 0)]
			  [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as array)", PrimaryILOffset = 69, MethodILOffset = 0)]
			  [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as array)", PrimaryILOffset = 83, MethodILOffset = 0)]
			  [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as array)", PrimaryILOffset = 87, MethodILOffset = 0)]
			  [RegressionOutcome(Outcome = ProofOutcome.True, Message = "assert is valid", PrimaryILOffset = 93, MethodILOffset = 0)]*/
	  void Test2(int[] arr, @IndexFor("#1") int i, @IndexFor("#1") int j)
	  {
		//Y: Contract.Requires(arr != null);
		  //Y: Contract.Requires(i >= 0);
		  //Y: Contract.Requires(i < arr.Length);
		  //Y: Contract.Requires(j >= 0);
		    //Y: Contract.Requires(j < arr.Length);
		    //N: Contract.Requires(LessThan(arr[i], arr[j]));
		  
	    int x = arr[i];
	    int y = arr[j];
	  }
}
