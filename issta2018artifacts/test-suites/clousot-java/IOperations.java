// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

import org.checkerframework.checker.index.qual.SameLen;
import org.checkerframework.common.value.qual.ArrayLen;

interface IOperation {
	 Class [] get_Types();
	 double Do(Object @SameLen("get_Types()")... operands);
	 
}

class OpA implements IOperation
{
    public Class @ArrayLen(2) [] get_Types()
    {
    	   //N: Contract.Ensures(Contract.Result<Type[]>() != null);
    	 //Y: Contract.Ensures(Contract.Result<Type[]>().Length == 2);
            
            return new Class [] { String.class, Double.class };
        
    }

/*    [ClousotRegressionTest]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "Lower bound access ok", PrimaryILOffset = 8, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "Upper bound access ok", PrimaryILOffset = 8, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "Lower bound access ok", PrimaryILOffset = 11, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "Upper bound access ok", PrimaryILOffset = 11, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "ensures is valid", PrimaryILOffset = 52, MethodILOffset = 36)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "ensures is valid", PrimaryILOffset = 77, MethodILOffset = 36)]*/

    
    
    public double Do(Object @SameLen("get_Types()") ... operands)
    {
    	//N: Contract.Requires(operands != null);
         //Y: Contract.Requires(operands.Length == Types.Length);
         //N: Contract.Ensures(Contract.Result<double>() >= 0);
         //N: Contract.Ensures(Contract.Result<double>() <= 1);
    	
        System.out.print(String.format("We have %s %ss\n", operands[1], operands[0]));
        return 0;
    }
}
