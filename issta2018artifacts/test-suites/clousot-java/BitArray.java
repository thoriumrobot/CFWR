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

import org.checkerframework.checker.index.qual.IndexFor;
import org.checkerframework.checker.index.qual.IndexOrHigh;
import org.checkerframework.checker.index.qual.LTEqLengthOf;
import org.checkerframework.checker.index.qual.NonNegative;
import org.checkerframework.checker.index.qual.Positive;
import org.checkerframework.common.value.qual.ArrayLenRange;
import org.checkerframework.common.value.qual.IntRange;
import org.checkerframework.dataflow.qual.Pure;


//From BitArray in the framework
public class BitArray {
    /*[ContractInvariantMethod]
    private void ObjectInvariant()
    {*/
      //N: Contract.Invariant(this.m_array != null);

      //Y: Contract.Invariant(this.m_length >= 0);
      //N: Contract.Invariant(this.m_array.Length <= this.m_length);
    /*}*/

	  private int[] m_array;
	    private @NonNegative int m_length;
	    private int _version;
	    
	    private static final int BitsPerInt32 = 32;
	    private static final int BytesPerInt32 = 4;
	    private static final int BitsPerByte = 8;
	    private static final int _ShrinkThreshold = 256;
	    
	    private void ArrayClear(int[] m_array2, int i, int j) {
			// TODO Auto-generated method stub
			
		}

	
    /*[ClousotRegressionTest("bitarray")]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as receiver)", PrimaryILOffset = 1, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as field receiver)", PrimaryILOffset = 32, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as field receiver)", PrimaryILOffset = 39, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as field receiver)", PrimaryILOffset = 71, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as array)", PrimaryILOffset = 76, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as field receiver)", PrimaryILOffset = 57, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as array)", PrimaryILOffset = 64, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as field receiver)", PrimaryILOffset = 82, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "Array creation : ok", PrimaryILOffset = 27, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "Lower bound access ok", PrimaryILOffset = 64, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "Upper bound access ok", PrimaryILOffset = 64, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "requires is valid", PrimaryILOffset = 4, MethodILOffset = 22)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "requires is valid", PrimaryILOffset = 16, MethodILOffset = 22)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "invariant is valid", PrimaryILOffset = 12, MethodILOffset = 87)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "invariant is valid", PrimaryILOffset = 29, MethodILOffset = 87)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "invariant is valid", PrimaryILOffset = 53, MethodILOffset = 87)]*/
	  public BitArray(@NonNegative int length, boolean defaultValue)
	    {
	      // F: old
	      //if (length < 0)
	      //{
	      //  throw new ArgumentOutOfRangeException();
	      //}
	      //Contract.EndContractBlock();

	      // F: new
	      

	      m_array = new int[GetArrayLength(length, BitsPerInt32)];
	      m_length = length;

	      int fillValue = defaultValue ? /*unchecked*/(((int)0xffffffff)) : 0;
	      for (int i = 0; i < m_array.length; i++)
	      {
	        m_array[i] = fillValue;
	      }

	      _version = 0;
	    }
    /*=========================================================================
    ** Allocates space to hold the bit values in bytes. bytes[0] represents
    ** bits 0 - 7, bytes[1] represents bits 8 - 15, etc. The LSB of each byte
    ** represents the lowest index value; bytes[0] & 1 represents bit 0,
    ** bytes[0] & 2 represents bit 1, bytes[0] & 4 represents bit 2, etc.
    **
    ** Exceptions: ArgumentException if bytes == null.
    =========================================================================*/
    /*
    [ClousotRegressionTest]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as receiver)", PrimaryILOffset = 1, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as array)", PrimaryILOffset = 26, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as array)", PrimaryILOffset = 43, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as field receiver)", PrimaryILOffset = 56, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as array)", PrimaryILOffset = 63, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as field receiver)", PrimaryILOffset = 67, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as array)", PrimaryILOffset = 148, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as field receiver)", PrimaryILOffset = 79, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as array)", PrimaryILOffset = 91, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as array)", PrimaryILOffset = 102, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as array)", PrimaryILOffset = 116, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as array)", PrimaryILOffset = 131, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as array)", PrimaryILOffset = 142, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as array)", PrimaryILOffset = 156, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as array)", PrimaryILOffset = 177, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as array)", PrimaryILOffset = 195, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as field receiver)", PrimaryILOffset = 223, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as array)", PrimaryILOffset = 233, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as array)", PrimaryILOffset = 243, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as field receiver)", PrimaryILOffset = 245, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as array)", PrimaryILOffset = 251, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as array)", PrimaryILOffset = 266, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as field receiver)", PrimaryILOffset = 282, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as array)", PrimaryILOffset = 288, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as array)", PrimaryILOffset = 301, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as field receiver)", PrimaryILOffset = 316, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "Array creation : ok", PrimaryILOffset = 51, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "Lower bound access ok", PrimaryILOffset = 91, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "Upper bound access ok", PrimaryILOffset = 91, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "Lower bound access ok", PrimaryILOffset = 102, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "Upper bound access ok", PrimaryILOffset = 102, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "Lower bound access ok", PrimaryILOffset = 116, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "Upper bound access ok", PrimaryILOffset = 116, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "Lower bound access ok", PrimaryILOffset = 131, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "Upper bound access ok", PrimaryILOffset = 131, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "Lower bound access ok", PrimaryILOffset = 142, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "Upper bound access ok", PrimaryILOffset = 142, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "Lower bound access ok", PrimaryILOffset = 233, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "Upper bound access ok", PrimaryILOffset = 233, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "Lower bound access ok", PrimaryILOffset = 243, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "Upper bound access ok", PrimaryILOffset = 243, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "Lower bound access ok", PrimaryILOffset = 251, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "Upper bound access ok", PrimaryILOffset = 251, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "Lower bound access ok", PrimaryILOffset = 266, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "Upper bound access ok", PrimaryILOffset = 266, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "Lower bound access ok", PrimaryILOffset = 288, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "Upper bound access ok", PrimaryILOffset = 288, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "Lower bound access ok", PrimaryILOffset = 301, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "Upper bound access ok", PrimaryILOffset = 301, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "requires is valid", PrimaryILOffset = 4, MethodILOffset = 46)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "requires is valid", PrimaryILOffset = 16, MethodILOffset = 46)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "assert is valid", PrimaryILOffset = 171, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "assert is valid", PrimaryILOffset = 189, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "invariant is valid", PrimaryILOffset = 12, MethodILOffset = 321)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "invariant is valid", PrimaryILOffset = 29, MethodILOffset = 321)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "invariant is valid", PrimaryILOffset = 53, MethodILOffset = 321)]*/
	  
	    public BitArray(byte[] bytes)
	    {
	      if (bytes == null)
	      {
	        throw new IllegalArgumentException("bytes");
	      }
	      //N: Contract.EndContractBlock();
	      // this value is chosen to prevent overflow when computing m_length.
	      // m_length is of type int32 and is exposed as a property, so 
	      // type of m_length can't be changed to accommodate.
	      if (bytes.length > Integer.MAX_VALUE / BitsPerByte)
	      {
	        throw new IllegalArgumentException();
	      }

	      m_array = new int[GetArrayLength(bytes.length, BytesPerInt32)];
	      m_length = bytes.length * BitsPerByte;

	      int i = 0;
	      int j = 0;
	      while (bytes.length - j >= 4)
	      {
	        m_array[i++] = (bytes[j] & 0xff) |
	                      ((bytes[j + 1] & 0xff) << 8) |
	                      ((bytes[j + 2] & 0xff) << 16) |
	                      ((bytes[j + 3] & 0xff) << 24);
	        j += 4;
	      }

	      //Y: Contract.Assert(bytes.length - j >= 0, "BitArray byteLength problem");
	      @NonNegative int _1 = bytes.length - j;
	      //Y: Contract.Assert(bytes.length - j < 4, "BitArray byteLength problem #2");
	      @IntRange(to = 3) int _2 = bytes.length - j;

	      switch (bytes.length - j)
	      {
	        case 3:
	          m_array[i] = ((bytes[j + 2] & 0xff) << 16);
	          
	        // fall through
	        case 2:
	          m_array[i] |= ((bytes[j + 1] & 0xff) << 8);
	          
	        // fall through
	        case 1:
	          m_array[i] |= (bytes[j] & 0xff);
	          break;
	      }

	      _version = 0;
	    }
	    
	    
	    /*[ClousotRegressionTest]
	    	    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as receiver)", PrimaryILOffset = 1, MethodILOffset = 0)]
	    	    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as array)", PrimaryILOffset = 27, MethodILOffset = 0)]
	    	    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as field receiver)", PrimaryILOffset = 41, MethodILOffset = 0)]
	    	    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as field receiver)", PrimaryILOffset = 47, MethodILOffset = 0)]
	    	    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as array)", PrimaryILOffset = 52, MethodILOffset = 0)]
	    	    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as array)", PrimaryILOffset = 58, MethodILOffset = 0)]
	    	    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as array)", PrimaryILOffset = 72, MethodILOffset = 0)]
	    	    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as field receiver)", PrimaryILOffset = 74, MethodILOffset = 0)]
	    	    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "Array creation : ok", PrimaryILOffset = 36, MethodILOffset = 0)]
	    	    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "requires is valid", PrimaryILOffset = 4, MethodILOffset = 31)]
	    	    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "requires is valid", PrimaryILOffset = 16, MethodILOffset = 31)]
	    	    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "assert is valid", PrimaryILOffset = 65, MethodILOffset = 0)]
	    	 [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"valid non-null reference (as array)",PrimaryILOffset=149,MethodILOffset=0)]
	    	 [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"valid non-null reference (as field receiver)",PrimaryILOffset=85,MethodILOffset=0)]
	    	 [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"valid non-null reference (as array)",PrimaryILOffset=90,MethodILOffset=0)]
	    	 [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"valid non-null reference (as array)",PrimaryILOffset=104,MethodILOffset=0)]
	    	 [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"valid non-null reference (as field receiver)",PrimaryILOffset=108,MethodILOffset=0)]
	    	 [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"valid non-null reference (as array)",PrimaryILOffset=117,MethodILOffset=0)]
	    	 [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"valid non-null reference (as field receiver)",PrimaryILOffset=155,MethodILOffset=0)]
	    	 [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"Lower bound access ok",PrimaryILOffset=104,MethodILOffset=0)]
	    	 [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"Upper bound access ok",PrimaryILOffset=104,MethodILOffset=0)]
	    	 [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"Lower bound access ok",PrimaryILOffset=117,MethodILOffset=0)]
	    	 [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"Upper bound access ok",PrimaryILOffset=117,MethodILOffset=0)]
	    	 [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"assert is valid",PrimaryILOffset=97,MethodILOffset=0)]
	    	 [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"invariant is valid",PrimaryILOffset=12,MethodILOffset=160)]
	    	 [RegressionOutcome(Outcome=ProofOutcome.True,Message=@"invariant is valid",PrimaryILOffset=29,MethodILOffset=160)]
	    	[RegressionOutcome(Outcome=ProofOutcome.True,Message=@"invariant is valid",PrimaryILOffset=53,MethodILOffset=160)]*/
	    public BitArray(boolean[] values)
	    {
	      if (values == null)
	      {
	        throw new IllegalArgumentException("values");
	      }
	    //N: Contract.EndContractBlock();

	      m_array = new int[GetArrayLength(values.length, BitsPerInt32)];

	      //N: Contract.Assert(m_array.length * 32 >= values.length);

	      m_length = values.length;

	      for (int i = 0; i < values.length; i++)
	      {
	        //N: Contract.Assert(i < m_array.length * 32);
	      			    
	       if (values[i])
	          m_array[i / 32] |= (1 << (i % 32));
	      }

	      _version = 0;
	    }
	    

	    /*=========================================================================
	    ** Allocates space to hold the bit values in values. values[0] represents
	    ** bits 0 - 31, values[1] represents bits 32 - 63, etc. The LSB of each
	    ** integer represents the lowest index value; values[0] & 1 represents bit
	    ** 0, values[0] & 2 represents bit 1, values[0] & 4 represents bit 2, etc.
	    **
	    ** Exceptions: ArgumentException if values == null.
	    =========================================================================*/
	    /*[ClousotRegressionTest("bitarray")]
	    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as array)", PrimaryILOffset = 19, MethodILOffset = 0)]
	    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as receiver)", PrimaryILOffset = 1, MethodILOffset = 0)]
	    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as array)", PrimaryILOffset = 38, MethodILOffset = 0)]
	    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as field receiver)", PrimaryILOffset = 45, MethodILOffset = 0)]
	    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as array)", PrimaryILOffset = 52, MethodILOffset = 0)]
	    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as field receiver)", PrimaryILOffset = 57, MethodILOffset = 0)]
	    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as field receiver)", PrimaryILOffset = 64, MethodILOffset = 0)]
	    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as array)", PrimaryILOffset = 70, MethodILOffset = 0)]
	    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as field receiver)", PrimaryILOffset = 79, MethodILOffset = 0)]
	    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as field receiver)", PrimaryILOffset = 85, MethodILOffset = 0)]
	    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as array)", PrimaryILOffset = 90, MethodILOffset = 0)]
	    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as field receiver)", PrimaryILOffset = 93, MethodILOffset = 0)]
	    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "Array creation : ok", PrimaryILOffset = 40, MethodILOffset = 0)]
	    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "assert is valid", PrimaryILOffset = 103, MethodILOffset = 0)]
	    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "invariant is valid", PrimaryILOffset = 12, MethodILOffset = 108)]
	    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "invariant is valid", PrimaryILOffset = 29, MethodILOffset = 108)]
	    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "invariant is valid", PrimaryILOffset = 53, MethodILOffset = 108)]*/
	    public BitArray(int @ArrayLenRange(to = Integer.MAX_VALUE / BitsPerInt32)[] values)
	    {
	      // F: old
	      //if (values == null)
	      //{
	      //  throw new ArgumentNullException("values");
	      //}
	      //Contract.EndContractBlock();

	      //// this value is chosen to prevent overflow when computing m_length
	      //if (values.length > Int32.MaxValue / BitsPerInt32)
	      //{
	      //  throw new ArgumentException();
	      //}

	      // F: new
	      //N: Contract.Requires(values != null);
	      //Y: Contract.Requires(values.length <= Int32.MaxValue / BitsPerInt32);

	      m_array = new int[values.length];
	      m_length = values.length * BitsPerInt32;

	      System.arraycopy(values, 0, m_array, 0, values.length);

	      _version = 0;

	      //N: Contract.Assert(this.m_array.length <= this.m_length);
	    }
	    /*=========================================================================
	     ** Allocates a new BitArray with the same length and bit values as bits.
	     **
	     ** Exceptions: ArgumentException if bits == null.
	     =========================================================================*/
	    /* [ClousotRegressionTest]
	     [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as receiver)", PrimaryILOffset = 1, MethodILOffset = 0)]
	     [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as field receiver)", PrimaryILOffset = 26, MethodILOffset = 0)]
	     [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as field receiver)", PrimaryILOffset = 43, MethodILOffset = 0)]
	     [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as field receiver)", PrimaryILOffset = 60, MethodILOffset = 0)]
	     [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as field receiver)", PrimaryILOffset = 75, MethodILOffset = 0)]
	     [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as array)", PrimaryILOffset = 80, MethodILOffset = 0)]
	     [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as field receiver)", PrimaryILOffset = 99, MethodILOffset = 0)]
	     [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as field receiver)", PrimaryILOffset = 106, MethodILOffset = 0)]
	     [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as field receiver)", PrimaryILOffset = 111, MethodILOffset = 0)]
	     [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as field receiver)", PrimaryILOffset = 117, MethodILOffset = 0)]
	     [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as field receiver)", PrimaryILOffset = 123, MethodILOffset = 0)]
	     [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as field receiver)", PrimaryILOffset = 136, MethodILOffset = 0)]
	     [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as field receiver)", PrimaryILOffset = 141, MethodILOffset = 0)]
	     [RegressionOutcome(Outcome = ProofOutcome.True, Message = "Array creation : ok", PrimaryILOffset = 94, MethodILOffset = 0)]
	     [RegressionOutcome(Outcome = ProofOutcome.True, Message = "requires is valid", PrimaryILOffset = 4, MethodILOffset = 67)]
	     [RegressionOutcome(Outcome = ProofOutcome.True, Message = "requires is valid", PrimaryILOffset = 16, MethodILOffset = 67)]
	     [RegressionOutcome(Outcome = ProofOutcome.True, Message = "invariant is valid", PrimaryILOffset = 12, MethodILOffset = 146)]
	     [RegressionOutcome(Outcome = ProofOutcome.True, Message = "invariant is valid", PrimaryILOffset = 29, MethodILOffset = 146)]
	     [RegressionOutcome(Outcome = ProofOutcome.True, Message = "invariant is valid", PrimaryILOffset = 53, MethodILOffset = 146)]*/
	    public BitArray(BitArray bits)
	    {
	      if (bits == null)
	      {
	        throw new IllegalArgumentException("bits");
	      }
	    //N: Contract.EndContractBlock();

	      // F: mine
	      //N: Contract.Assume(bits.m_array != null);
	      //N: Contract.Assume(bits.m_length >= 0);

	      int arrayLength = GetArrayLength(bits.m_length, BitsPerInt32);

	      //N: Contract.Assume(arrayLength <= bits.m_array.length);

	      m_array = new int[arrayLength];
	      m_length = bits.m_length;

	      // :: error: (argument.type.incompatible)
	      System.arraycopy(bits.m_array,0, m_array, 0, arrayLength);

	      _version = bits._version;
	    }
	    /*[ClousotRegressionTest("bitarray")]*/
	    public boolean get_Item(@NonNegative int index)
	    	    {
	    	      
	    	        // F: Pushing the precondition of Get
	    	        //Y: Contract.Requires(index >= 0);
	    	        //N: Contract.Requires(index < Length);

	    	        return Get(index);
	    	      }
	    	      public void set_Item(@NonNegative int index, boolean value)
	    	      {
	    	        // F: Pushing the precondition of Get
	    	        //Y: Contract.Requires(index >= 0);
	    	        //N: Contract.Requires(index < Length);

	    	        Set(index, value);
	    	      }
	    	    
	    	      /*=========================================================================
	    	       ** Returns the bit value at position index.
	    	       **
	    	       ** Exceptions: ArgumentOutOfRangeException if index < 0 or
	    	       **             index >= GetLength().
	    	       =========================================================================*/
	    	       /*[ClousotRegressionTest("bitarray")]
	    	       [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as receiver)", PrimaryILOffset = 14, MethodILOffset = 0)]
	    	       [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as field receiver)", PrimaryILOffset = 31, MethodILOffset = 0)]
	    	       [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as array)", PrimaryILOffset = 36, MethodILOffset = 0)]
	    	       [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as field receiver)", PrimaryILOffset = 46, MethodILOffset = 0)]
	    	       [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as array)", PrimaryILOffset = 55, MethodILOffset = 0)]
	    	       [RegressionOutcome(Outcome = ProofOutcome.True, Message = "Lower bound access ok", PrimaryILOffset = 55, MethodILOffset = 0)]
	    	       [RegressionOutcome(Outcome = ProofOutcome.True, Message = "Upper bound access ok", PrimaryILOffset = 55, MethodILOffset = 0)]
	    	       [RegressionOutcome(Outcome = ProofOutcome.True, Message = "invariant is valid", PrimaryILOffset = 12, MethodILOffset = 72)]
	    	       [RegressionOutcome(Outcome = ProofOutcome.True, Message = "invariant is valid", PrimaryILOffset = 29, MethodILOffset = 72)]
	    	       [RegressionOutcome(Outcome = ProofOutcome.True, Message = "invariant is valid", PrimaryILOffset = 53, MethodILOffset = 72)]*/
	    public boolean Get(@NonNegative int index)
	    {
	      // F: Rewritten Precondition

	      // F: Old
	      //if (index < 0 || index >= Length)
	      //{
	      //  throw new ArgumentOutOfRangeException();
	      //}
	      //Contract.EndContractBlock();

	      // F: new
	      //Y: Contract.Requires(index >= 0);
	      //N: Contract.Requires(index < Length);

	      // F: mine, as it is difficult to express the relationship between index, m_length and m_array.length
	      //N: Contract.Assume(index / 32 < this.m_array.length);

	    	// :: error: (array.access.unsafe.high.range)
	      return (m_array[index / 32] & (1 << (index % 32))) != 0;
	    }
	    /*=========================================================================
	     ** Sets the bit value at position index to value.
	     **
	     ** Exceptions: ArgumentOutOfRangeException if index < 0 or
	     **             index >= GetLength().
	     =========================================================================*/
	     /*[ClousotRegressionTest("bitarray")]
	     [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as receiver)", PrimaryILOffset = 14, MethodILOffset = 0)]
	     [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as field receiver)", PrimaryILOffset = 31, MethodILOffset = 0)]
	     [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as array)", PrimaryILOffset = 36, MethodILOffset = 0)]
	     [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as field receiver)", PrimaryILOffset = 49, MethodILOffset = 0)]
	     [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as array)", PrimaryILOffset = 58, MethodILOffset = 0)]
	     [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as field receiver)", PrimaryILOffset = 87, MethodILOffset = 0)]
	     [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as array)", PrimaryILOffset = 96, MethodILOffset = 0)]
	     [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as field receiver)", PrimaryILOffset = 125, MethodILOffset = 0)]
	     [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as field receiver)", PrimaryILOffset = 132, MethodILOffset = 0)]
	     [RegressionOutcome(Outcome = ProofOutcome.True, Message = "Lower bound access ok", PrimaryILOffset = 58, MethodILOffset = 0)]
	     [RegressionOutcome(Outcome = ProofOutcome.True, Message = "Upper bound access ok", PrimaryILOffset = 58, MethodILOffset = 0)]
	     [RegressionOutcome(Outcome = ProofOutcome.True, Message = "Lower bound access ok", PrimaryILOffset = 96, MethodILOffset = 0)]
	     [RegressionOutcome(Outcome = ProofOutcome.True, Message = "Upper bound access ok", PrimaryILOffset = 96, MethodILOffset = 0)]
	     [RegressionOutcome(Outcome = ProofOutcome.True, Message = "invariant is valid", PrimaryILOffset = 12, MethodILOffset = 137)]
	     [RegressionOutcome(Outcome = ProofOutcome.True, Message = "invariant is valid", PrimaryILOffset = 29, MethodILOffset = 137)]
	     [RegressionOutcome(Outcome = ProofOutcome.True, Message = "invariant is valid", PrimaryILOffset = 53, MethodILOffset = 137)]*/

	    public void Set(@NonNegative int index, boolean value)
	    {
	      // F: Old
	      //if (index < 0 || index >= Length)
	      //{
	      //  throw new ArgumentOutOfRangeException();
	      //}
	      //Contract.EndContractBlock();

	      // F: New
	      //Y: Contract.Requires(index >= 0);
	      //N: Contract.Requires(index < Length);

	      // F: mine, as it is difficult to express the relationship between index, m_length and m_array.length
	      //N: Contract.Assume(index / 32 < this.m_array.length);

	      if (value)
	      {
	    	  // :: error: (array.access.unsafe.high.range)
	        m_array[index / 32] |= (1 << (index % 32));
	      }
	      else
	      {
	    	  // :: error: (array.access.unsafe.high.range)
	        m_array[index / 32] &= ~(1 << (index % 32));
	      }

	      _version++;
	    }
	    /*=========================================================================
	     ** Sets all the bit values to value.
	     =========================================================================*/
	     /*[ClousotRegressionTest("bitarray")]
	     [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as field receiver)", PrimaryILOffset = 9, MethodILOffset = 0)]
	     [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as field receiver)", PrimaryILOffset = 24, MethodILOffset = 0)]
	     [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as array)", PrimaryILOffset = 29, MethodILOffset = 0)]
	     [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as field receiver)", PrimaryILOffset = 46, MethodILOffset = 0)]
	     [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as array)", PrimaryILOffset = 53, MethodILOffset = 0)]
	     [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as field receiver)", PrimaryILOffset = 64, MethodILOffset = 0)]
	     [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as field receiver)", PrimaryILOffset = 71, MethodILOffset = 0)]
	     [RegressionOutcome(Outcome = ProofOutcome.True, Message = "Lower bound access ok", PrimaryILOffset = 53, MethodILOffset = 0)]
	     [RegressionOutcome(Outcome = ProofOutcome.True, Message = "Upper bound access ok", PrimaryILOffset = 53, MethodILOffset = 0)]
	     [RegressionOutcome(Outcome = ProofOutcome.True, Message = "requires is valid", PrimaryILOffset = 4, MethodILOffset = 16)]
	     [RegressionOutcome(Outcome = ProofOutcome.True, Message = "requires is valid", PrimaryILOffset = 16, MethodILOffset = 16)]
	     [RegressionOutcome(Outcome = ProofOutcome.True, Message = "invariant is valid", PrimaryILOffset = 12, MethodILOffset = 76)]
	     [RegressionOutcome(Outcome = ProofOutcome.True, Message = "invariant is valid", PrimaryILOffset = 29, MethodILOffset = 76)]
	     [RegressionOutcome(Outcome = ProofOutcome.True, Message = "invariant is valid", PrimaryILOffset = 53, MethodILOffset = 76)]*/
	    public void SetAll(boolean value)
	    {
	      int fillValue = value ? /*unchecked*/(((int)0xffffffff)) : 0;
	      int ints = GetArrayLength(m_length, BitsPerInt32);

	      // f: Added as there is no simple way to express the relation between m_length, m_array.length and ints
	      //N: Contract.Assume(ints <= m_array.length);

	      //Contract.Assert(this.m_array.length * 32 >= this.m_length);

	      for (int i = 0; i < ints; i++)
	      {
	    	  // :: error: (array.access.unsafe.high.range)
	        m_array[i] = fillValue;
	      }

	      _version++;
	    }
	    

	    /*=========================================================================
	    ** Returns a reference to the current instance ANDed with value.
	    **
	    ** Exceptions: ArgumentException if value == null or
	    **             value.Length != this.Length.
	    =========================================================================*/
	    /*[ClousotRegressionTest("bitarray")]
	    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as receiver)", PrimaryILOffset = 13, MethodILOffset = 0)]
	    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as receiver)", PrimaryILOffset = 19, MethodILOffset = 0)]
	    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as field receiver)", PrimaryILOffset = 32, MethodILOffset = 0)]
	    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as field receiver)", PrimaryILOffset = 47, MethodILOffset = 0)]
	    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as array)", PrimaryILOffset = 52, MethodILOffset = 0)]
	    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as field receiver)", PrimaryILOffset = 66, MethodILOffset = 0)]
	    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as array)", PrimaryILOffset = 71, MethodILOffset = 0)]
	    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as field receiver)", PrimaryILOffset = 88, MethodILOffset = 0)]
	    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as array)", PrimaryILOffset = 94, MethodILOffset = 0)]
	    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as field receiver)", PrimaryILOffset = 106, MethodILOffset = 0)]
	    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as array)", PrimaryILOffset = 112, MethodILOffset = 0)]
	    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as field receiver)", PrimaryILOffset = 129, MethodILOffset = 0)]
	    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as field receiver)", PrimaryILOffset = 136, MethodILOffset = 0)]
	    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "Lower bound access ok", PrimaryILOffset = 94, MethodILOffset = 0)]
	    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "Upper bound access ok", PrimaryILOffset = 94, MethodILOffset = 0)]
	    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "Lower bound access ok", PrimaryILOffset = 112, MethodILOffset = 0)]
	    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "Upper bound access ok", PrimaryILOffset = 112, MethodILOffset = 0)]
	    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "requires is valid", PrimaryILOffset = 4, MethodILOffset = 39)]
	    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "requires is valid", PrimaryILOffset = 16, MethodILOffset = 39)]
	    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "invariant is valid", PrimaryILOffset = 12, MethodILOffset = 142)]
	    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "invariant is valid", PrimaryILOffset = 29, MethodILOffset = 142)]
	    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "invariant is valid", PrimaryILOffset = 53, MethodILOffset = 142)]*/
	    public BitArray And(BitArray value)
	    {
	      // f: old
	      //if (value == null)
	      //  throw new ArgumentNullException("value");
	      //if (Length != value.length)
	      //  throw new ArgumentException();
	      //Contract.EndContractBlock();

	      // f: new
	      //N: Contract.Requires(value != null);
	      //N: Contract.Requires(Length == value.length);

	      int ints = GetArrayLength(m_length, BitsPerInt32);

	      // f: Added as there is no simple way to express the relation between m_length, m_array.length and ints
	      //N: Contract.Assume(ints <= m_array.length);
	      //N: Contract.Assume(ints <= value.m_array.length);

	      for (int i = 0; i < ints; i++)
	      {
	    	  // :: error: (array.access.unsafe.high.range) :: error: (array.access.unsafe.high.range)
	        m_array[i] &= value.m_array[i];
	      }

	      _version++;
	      return this;
	    }

	    /*=========================================================================
	    ** Returns a reference to the current instance ORed with value.
	    **
	    ** Exceptions: ArgumentException if value == null or
	    **             value.Length != this.Length.
	    =========================================================================*/
	    /*[ClousotRegressionTest("bitarray")]
	    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as receiver)", PrimaryILOffset = 15, MethodILOffset = 0)]
	    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as receiver)", PrimaryILOffset = 21, MethodILOffset = 0)]
	    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as field receiver)", PrimaryILOffset = 40, MethodILOffset = 0)]
	    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as field receiver)", PrimaryILOffset = 55, MethodILOffset = 0)]
	    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as array)", PrimaryILOffset = 60, MethodILOffset = 0)]
	    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as field receiver)", PrimaryILOffset = 74, MethodILOffset = 0)]
	    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as array)", PrimaryILOffset = 79, MethodILOffset = 0)]
	    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as field receiver)", PrimaryILOffset = 96, MethodILOffset = 0)]
	    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as array)", PrimaryILOffset = 102, MethodILOffset = 0)]
	    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as field receiver)", PrimaryILOffset = 114, MethodILOffset = 0)]
	    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as array)", PrimaryILOffset = 120, MethodILOffset = 0)]
	    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as field receiver)", PrimaryILOffset = 137, MethodILOffset = 0)]
	    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as field receiver)", PrimaryILOffset = 144, MethodILOffset = 0)]
	    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "Lower bound access ok", PrimaryILOffset = 102, MethodILOffset = 0)]
	    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "Upper bound access ok", PrimaryILOffset = 102, MethodILOffset = 0)]
	    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "Lower bound access ok", PrimaryILOffset = 120, MethodILOffset = 0)]
	    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "Upper bound access ok", PrimaryILOffset = 120, MethodILOffset = 0)]
	    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "requires is valid", PrimaryILOffset = 4, MethodILOffset = 47)]
	    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "requires is valid", PrimaryILOffset = 16, MethodILOffset = 47)]
	    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "invariant is valid", PrimaryILOffset = 12, MethodILOffset = 150)]
	    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "invariant is valid", PrimaryILOffset = 29, MethodILOffset = 150)]
	    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "invariant is valid", PrimaryILOffset = 53, MethodILOffset = 150)]*/
	    public BitArray Or(BitArray value)
	    {
	      if (value == null)
	        throw new IllegalArgumentException("value");
	      if (get_Length() != value.get_Length())
	        throw new IllegalArgumentException();
	      //N: Contract.EndContractBlock();

	      int ints = GetArrayLength(m_length, BitsPerInt32);

	      // f: Added as there is no simple way to express the relation between m_length, m_array.length and ints
	      //N: Contract.Assume(ints <= m_array.length);
	      //N: Contract.Assume(ints <= value.m_array.length);

	      for (int i = 0; i < ints; i++)
	      {
	    	  // :: error: (array.access.unsafe.high.range) :: error: (array.access.unsafe.high.range)
	        m_array[i] |= value.m_array[i];
	      }

	      _version++;
	      return this;
	    }
	    /*=========================================================================
	     ** Returns a reference to the current instance XORed with value.
	     **
	     ** Exceptions: ArgumentException if value == null or
	     **             value.Length != this.Length.
	     =========================================================================*/
	     /*[ClousotRegressionTest("bitarray")]
	     [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as receiver)", PrimaryILOffset = 15, MethodILOffset = 0)]
	     [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as receiver)", PrimaryILOffset = 21, MethodILOffset = 0)]
	     [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as field receiver)", PrimaryILOffset = 40, MethodILOffset = 0)]
	     [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as field receiver)", PrimaryILOffset = 55, MethodILOffset = 0)]
	     [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as array)", PrimaryILOffset = 60, MethodILOffset = 0)]
	     [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as field receiver)", PrimaryILOffset = 74, MethodILOffset = 0)]
	     [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as array)", PrimaryILOffset = 79, MethodILOffset = 0)]
	     [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as field receiver)", PrimaryILOffset = 96, MethodILOffset = 0)]
	     [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as array)", PrimaryILOffset = 102, MethodILOffset = 0)]
	     [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as field receiver)", PrimaryILOffset = 114, MethodILOffset = 0)]
	     [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as array)", PrimaryILOffset = 120, MethodILOffset = 0)]
	     [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as field receiver)", PrimaryILOffset = 137, MethodILOffset = 0)]
	     [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as field receiver)", PrimaryILOffset = 144, MethodILOffset = 0)]
	     [RegressionOutcome(Outcome = ProofOutcome.True, Message = "Lower bound access ok", PrimaryILOffset = 102, MethodILOffset = 0)]
	     [RegressionOutcome(Outcome = ProofOutcome.True, Message = "Upper bound access ok", PrimaryILOffset = 102, MethodILOffset = 0)]
	     [RegressionOutcome(Outcome = ProofOutcome.True, Message = "Lower bound access ok", PrimaryILOffset = 120, MethodILOffset = 0)]
	     [RegressionOutcome(Outcome = ProofOutcome.True, Message = "Upper bound access ok", PrimaryILOffset = 120, MethodILOffset = 0)]
	     [RegressionOutcome(Outcome = ProofOutcome.True, Message = "requires is valid", PrimaryILOffset = 4, MethodILOffset = 47)]
	     [RegressionOutcome(Outcome = ProofOutcome.True, Message = "requires is valid", PrimaryILOffset = 16, MethodILOffset = 47)]
	     [RegressionOutcome(Outcome = ProofOutcome.True, Message = "invariant is valid", PrimaryILOffset = 12, MethodILOffset = 150)]
	     [RegressionOutcome(Outcome = ProofOutcome.True, Message = "invariant is valid", PrimaryILOffset = 29, MethodILOffset = 150)]
	     [RegressionOutcome(Outcome = ProofOutcome.True, Message = "invariant is valid", PrimaryILOffset = 53, MethodILOffset = 150)]*/
	    public BitArray Xor(BitArray value)
	    {
	      if (value == null)
	        throw new IllegalArgumentException("value");
	      if (get_Length() != value.get_Length())
	        throw new IllegalArgumentException();
	      //N: Contract.EndContractBlock();

	      int ints = GetArrayLength(m_length, BitsPerInt32);

	      // f: Added as there is no simple way to express the relation between m_length, m_array.length and ints
	      //N: Contract.Assume(ints <= m_array.length);
	      //N: Contract.Assume(ints <= value.m_array.length);

	      for (int i = 0; i < ints; i++)
	      {
	    	 // :: error: (array.access.unsafe.high.range) :: error: (array.access.unsafe.high.range)
	        m_array[i] ^= value.m_array[i];
	      }

	      _version++;
	      return this;
	    }
	    /*=========================================================================
	     ** Inverts all the bit values. On/true bit values are converted to
	     ** off/false. Off/false bit values are turned on/true. The current instance
	     ** is updated and returned.
	     =========================================================================*/
	     /*[ClousotRegressionTest("bitarray")]
	     [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as field receiver)", PrimaryILOffset = 1, MethodILOffset = 0)]
	     [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as field receiver)", PrimaryILOffset = 16, MethodILOffset = 0)]
	     [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as array)", PrimaryILOffset = 21, MethodILOffset = 0)]
	     [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as field receiver)", PrimaryILOffset = 38, MethodILOffset = 0)]
	     [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as field receiver)", PrimaryILOffset = 45, MethodILOffset = 0)]
	     [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as array)", PrimaryILOffset = 51, MethodILOffset = 0)]
	     [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as array)", PrimaryILOffset = 53, MethodILOffset = 0)]
	     [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as field receiver)", PrimaryILOffset = 64, MethodILOffset = 0)]
	     [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as field receiver)", PrimaryILOffset = 71, MethodILOffset = 0)]
	     [RegressionOutcome(Outcome = ProofOutcome.True, Message = "Lower bound access ok", PrimaryILOffset = 51, MethodILOffset = 0)]
	     [RegressionOutcome(Outcome = ProofOutcome.True, Message = "Upper bound access ok", PrimaryILOffset = 51, MethodILOffset = 0)]
	     [RegressionOutcome(Outcome = ProofOutcome.True, Message = "Lower bound access ok", PrimaryILOffset = 53, MethodILOffset = 0)]
	     [RegressionOutcome(Outcome = ProofOutcome.True, Message = "Upper bound access ok", PrimaryILOffset = 53, MethodILOffset = 0)]
	     [RegressionOutcome(Outcome = ProofOutcome.True, Message = "requires is valid", PrimaryILOffset = 4, MethodILOffset = 8)]
	     [RegressionOutcome(Outcome = ProofOutcome.True, Message = "requires is valid", PrimaryILOffset = 16, MethodILOffset = 8)]
	     [RegressionOutcome(Outcome = ProofOutcome.True, Message = "invariant is valid", PrimaryILOffset = 12, MethodILOffset = 77)]
	     [RegressionOutcome(Outcome = ProofOutcome.True, Message = "invariant is valid", PrimaryILOffset = 29, MethodILOffset = 77)]
	     [RegressionOutcome(Outcome = ProofOutcome.True, Message = "invariant is valid", PrimaryILOffset = 53, MethodILOffset = 77)]*/
	    public BitArray Not()
	    {
	      int ints = GetArrayLength(m_length, BitsPerInt32);


	      // f: Added as there is no simple way to express the relation between m_length, m_array.length and ints
	      //Y: Contract.Assume(ints <= m_array.length);
	      @SuppressWarnings("index")
	      @LTEqLengthOf("m_array") int ints_1 = ints;

	      for (int i = 0; i < ints_1; i++)
	      {
	        m_array[i] = ~m_array[i];
	      }

	      _version++;
	      return this;
	    }
	    /*[ClousotRegressionTest("bitarray")]
	    	      [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as field receiver)", PrimaryILOffset = 19, MethodILOffset = 0)]
	    	      [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as field receiver)", PrimaryILOffset = 6, MethodILOffset = 24)]
	    	      [RegressionOutcome(Outcome = ProofOutcome.True, Message = "ensures is valid", PrimaryILOffset = 13, MethodILOffset = 24)]*/

	    public int get_Length()
	      {
	        // F: Added
	        //N: Contract.Ensures(Contract.Result<int>() == this.m_length);

	        return m_length;
	      }
	    /*[ClousotRegressionTest("bitarray")]
	    	      [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as field receiver)", PrimaryILOffset = 23, MethodILOffset = 0)]
	    	      [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as array)", PrimaryILOffset = 28, MethodILOffset = 0)]
	    	      [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as field receiver)", PrimaryILOffset = 40, MethodILOffset = 0)]
	    	      [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as array)", PrimaryILOffset = 45, MethodILOffset = 0)]
	    	      [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as field receiver)", PrimaryILOffset = 57, MethodILOffset = 0)]
	    	      [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as field receiver)", PrimaryILOffset = 65, MethodILOffset = 0)]
	    	      [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as array)", PrimaryILOffset = 70, MethodILOffset = 0)]
	    	      [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as field receiver)", PrimaryILOffset = 78, MethodILOffset = 0)]
	    	      [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as array)", PrimaryILOffset = 83, MethodILOffset = 0)]
	    	      [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as field receiver)", PrimaryILOffset = 92, MethodILOffset = 0)]
	    	      [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as field receiver)", PrimaryILOffset = 99, MethodILOffset = 0)]
	    	      [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as field receiver)", PrimaryILOffset = 107, MethodILOffset = 0)]
	    	      [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as field receiver)", PrimaryILOffset = 136, MethodILOffset = 0)]
	    	      [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as array)", PrimaryILOffset = 141, MethodILOffset = 0)]
	    	      [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as field receiver)", PrimaryILOffset = 160, MethodILOffset = 0)]
	    	      [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as field receiver)", PrimaryILOffset = 174, MethodILOffset = 0)]
	    	      [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as array)", PrimaryILOffset = 180, MethodILOffset = 0)]
	    	      [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as field receiver)", PrimaryILOffset = 206, MethodILOffset = 0)]
	    	      [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as field receiver)", PrimaryILOffset = 226, MethodILOffset = 0)]
	    	      [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as field receiver)", PrimaryILOffset = 233, MethodILOffset = 0)]
	    	      [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as field receiver)", PrimaryILOffset = 240, MethodILOffset = 0)]
	    	      [RegressionOutcome(Outcome = ProofOutcome.True, Message = "Array creation : ok", PrimaryILOffset = 50, MethodILOffset = 0)]
	    	      [RegressionOutcome(Outcome = ProofOutcome.True, Message = "Lower bound access ok", PrimaryILOffset = 180, MethodILOffset = 0)]
	    	      [RegressionOutcome(Outcome = ProofOutcome.True, Message = "Upper bound access ok", PrimaryILOffset = 180, MethodILOffset = 0)]
	    	      [RegressionOutcome(Outcome = ProofOutcome.True, Message = "requires is valid", PrimaryILOffset = 4, MethodILOffset = 15)]
	    	      [RegressionOutcome(Outcome = ProofOutcome.True, Message = "requires is valid", PrimaryILOffset = 16, MethodILOffset = 15)]
	    	      [RegressionOutcome(Outcome = ProofOutcome.True, Message = "requires is valid", PrimaryILOffset = 4, MethodILOffset = 114)]
	    	      [RegressionOutcome(Outcome = ProofOutcome.True, Message = "requires is valid", PrimaryILOffset = 16, MethodILOffset = 114)]
	    	      [RegressionOutcome(Outcome = ProofOutcome.True, Message = "invariant is valid", PrimaryILOffset = 12, MethodILOffset = 245)]
	    	      [RegressionOutcome(Outcome = ProofOutcome.True, Message = "invariant is valid", PrimaryILOffset = 29, MethodILOffset = 245)]
	    	      [RegressionOutcome(Outcome = ProofOutcome.Top, Message = @"invariant unproven: this.m_array.Length <= this.m_length", PrimaryILOffset = 53, MethodILOffset = 245)]*/
	    	  
	    public void set_Length(@NonNegative int value) {
	    	 {
	    	        //if (value < 0)
	    	        //{
	    	        //  throw new ArgumentOutOfRangeException();
	    	        //}
	    	        //Contract.EndContractBlock();

	    	        //Y: Contract.Requires(value >= 0);

	    	        int newints = GetArrayLength(value, BitsPerInt32);
	    	        if (newints > m_array.length || newints + _ShrinkThreshold < m_array.length)
	    	        {
	    	          // grow or shrink (if wasting more than _ShrinkThreshold ints)
	    	          int[] newarray = new int[newints];
	    	          System.arraycopy(m_array,0, newarray,0, newints > m_array.length ? m_array.length : newints);
	    	          m_array = newarray;
	    	        }

	    	        // F: is it true??? I do not think so, the code below will set the invariant?
	    	        // Contract.Assert(this.m_array.length <= this.m_length);

	    	        if (value > m_length)
	    	        {
	    	          // clear high bit values in the last int
	    	          int last = GetArrayLength(m_length, BitsPerInt32) - 1;

	    	          // F: Those are mine. Those are mathematical properties of GetArrayLength
	    	          //Y: Contract.Assume(last >= 0);
	    	          //Y: Contract.Assume(last < m_array.length);
	    	          //N: Contract.Assume(newints > last);
	    	          @SuppressWarnings("index")
	    	          @IndexFor("m_array") int last_1 = last;

	    	          int bits = m_length % 32;
	    	          if (bits > 0)
	    	          {
	    	            m_array[last_1] &= (1 << bits) - 1;
	    	          }

	    	          // clear remaining int values
	    	          ArrayClear(m_array, last_1 + 1, newints - last - 1);
	    	        }

	    	        m_length = value;
	    	        _version++;
	    	      }
	    }
	  
		/* // ICollection implementation
    [ClousotRegressionTest("bitarray")]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as receiver)", PrimaryILOffset = 25, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as field receiver)", PrimaryILOffset = 48, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as receiver)", PrimaryILOffset = 62, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as field receiver)", PrimaryILOffset = 78, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as field receiver)", PrimaryILOffset = 91, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as array)", PrimaryILOffset = 96, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as field receiver)", PrimaryILOffset = 109, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as field receiver)", PrimaryILOffset = 118, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as field receiver)", PrimaryILOffset = 145, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as receiver)", PrimaryILOffset = 158, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as field receiver)", PrimaryILOffset = 175, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as array)", PrimaryILOffset = 180, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as field receiver)", PrimaryILOffset = 225, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as array)", PrimaryILOffset = 233, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as array)", PrimaryILOffset = 250, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as receiver)", PrimaryILOffset = 269, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as field receiver)", PrimaryILOffset = 277, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as field receiver)", PrimaryILOffset = 298, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as array)", PrimaryILOffset = 303, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as field receiver)", PrimaryILOffset = 309, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as field receiver)", PrimaryILOffset = 373, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as field receiver)", PrimaryILOffset = 335, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as array)", PrimaryILOffset = 345, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as array)", PrimaryILOffset = 363, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "Lower bound access ok", PrimaryILOffset = 233, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "Upper bound access ok", PrimaryILOffset = 233, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "Lower bound access ok", PrimaryILOffset = 250, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "Upper bound access ok", PrimaryILOffset = 250, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "Lower bound access ok", PrimaryILOffset = 345, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.Top,  Message = "Array access might be above the upper bound", PrimaryILOffset = 345, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "Lower bound access ok", PrimaryILOffset = 363, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "Upper bound access ok", PrimaryILOffset = 363, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "requires is valid", PrimaryILOffset = 4, MethodILOffset = 55)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "requires is valid", PrimaryILOffset = 16, MethodILOffset = 55)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "requires is valid", PrimaryILOffset = 4, MethodILOffset = 85)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "requires is valid", PrimaryILOffset = 16, MethodILOffset = 85)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "requires is valid", PrimaryILOffset = 4, MethodILOffset = 125)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "requires is valid", PrimaryILOffset = 16, MethodILOffset = 125)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "invariant is valid", PrimaryILOffset = 12, MethodILOffset = 135)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "invariant is valid", PrimaryILOffset = 29, MethodILOffset = 135)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "invariant is valid", PrimaryILOffset = 53, MethodILOffset = 135)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "requires is valid", PrimaryILOffset = 4, MethodILOffset = 151)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "requires is valid", PrimaryILOffset = 16, MethodILOffset = 151)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "assert is valid", PrimaryILOffset = 215, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "invariant is valid", PrimaryILOffset = 12, MethodILOffset = 259)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "invariant is valid", PrimaryILOffset = 29, MethodILOffset = 259)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "invariant is valid", PrimaryILOffset = 53, MethodILOffset = 259)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "invariant is valid", PrimaryILOffset = 12, MethodILOffset = 380)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "invariant is valid", PrimaryILOffset = 29, MethodILOffset = 380)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "invariant is valid", PrimaryILOffset = 53, MethodILOffset = 380)]
*/
	    public void CopyTo(Object[] array, @NonNegative int index)
	    {
	      //if (array == null)
	      //  throw new ArgumentNullException("array");

	      //if (index < 0)
	      //  throw new ArgumentOutOfRangeException();

	      //if (array.Rank != 1)
	      //  throw new ArgumentException();

	      //Contract.EndContractBlock();

	      // F: just rewrote the ones above
	      //N: Contract.Requires(array != null);
	      //Y: Contract.Requires(index >= 0);
	      //N: Contract.Requires(array.Rank == 1);

	      if (array instanceof Integer[])
	      {
	        // F: Those looks like pre-conditions, but they are not as they involve some private state...
	        //N: Contract.Assume(index + GetArrayLength(m_length, BitsPerInt32) <= array.length);
	        //N: Contract.Assume(GetArrayLength(m_length, BitsPerInt32) <= m_array.length);
	    	// :: error: (argument.type.incompatible)
	        System.arraycopy(m_array, 0, array, index, GetArrayLength(m_length, BitsPerInt32));
	      }
	      else if (array instanceof Byte[])
	      {
	        int arrayLength = GetArrayLength(m_length, BitsPerByte);
	        if ((array.length - index) < arrayLength)
	          throw new IllegalArgumentException();

	        // F: Mine as the relation is difficult to express
	        //N: Contract.Assume(m_array.length * 4 >= arrayLength);

	        Byte[] b = (Byte[])array;
	        for (int i = 0; i < arrayLength; i++)
	        {
	          //Y: Contract.Assert(i / 4 >= 0);    // ok, can prove it
	        	@NonNegative int _1 = i/4;

	        	// :: error: (array.access.unsafe.high)
	          b[index + i] = (byte)((m_array[i / 4] >> ((i % 4) * 8)) & 0x000000FF); // Shift to bring the required byte to LSB, then mask
	        }
	      }
	      else if (array instanceof Boolean[])
	      {
	        if (array.length - index < m_length)
	          throw new IllegalArgumentException();

	        Boolean[] b = (Boolean[])array;

	        // F: mine
	        //N: Contract.Assume(m_array.length * 32 >= m_length);

	        for (int i = 0; i < m_length; i++)
	        {
	        	// :: error: (array.access.unsafe.high) :: error: (array.access.unsafe.high)
	          b[index + i] = ((m_array[i / 32] >> (i % 32)) & 0x00000001) != 0;
	        }
	      }
	      else
	        throw new IllegalArgumentException();
	    }
	    /*[ClousotRegressionTest("bitarray")]
	    	      [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as field receiver)", PrimaryILOffset = 35, MethodILOffset = 0)]
	    	      [RegressionOutcome(Outcome = ProofOutcome.True, Message = "valid non-null reference (as field receiver)", PrimaryILOffset = 6, MethodILOffset = 40)]
	    	      [RegressionOutcome(Outcome = ProofOutcome.True, Message = "ensures is valid", PrimaryILOffset = 13, MethodILOffset = 40)]
	    	      [RegressionOutcome(Outcome = ProofOutcome.True, Message = "ensures is valid", PrimaryILOffset = 29, MethodILOffset = 40)]*/
	    	  
	    @NonNegative int get_Count()
	      {
	        // F: Added
	        //N: Contract.Ensures(Contract.Result<int>() == this.m_length);
	        //Y: Contract.Ensures(Contract.Result<int>() >= 0);

	        return m_length;
	      }
	    
	    
	 
	    	    /*[ClousotRegressionTest("bitarray")]
	    	    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "ensures is valid", PrimaryILOffset = 32, MethodILOffset = 76)]
	    	    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "ensures is valid", PrimaryILOffset = 48, MethodILOffset = 76)]
	    	    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "ensures is valid", PrimaryILOffset = 66, MethodILOffset = 76)]
	    	    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "ensures is valid", PrimaryILOffset = 32, MethodILOffset = 84)]
	    	    [RegressionOutcome(Outcome = ProofOutcome.Top, Message = @"ensures unproven: Contract.Result<int>() <= n. The static checker determined that the condition '((n - 1) / div + 1) <= n' should hold on entry. Nevertheless, the condition may be too strong for the callers. If you think it is ok, add a precondition to document it: Contract.Requires(((n - 1) / div + 1) <= n);", PrimaryILOffset = 48, MethodILOffset = 84)]
	    	    //[RegressionOutcome(Outcome = ProofOutcome.True, Message = "ensures is valid", PrimaryILOffset = 66, MethodILOffset = 84)]
	    	    [RegressionOutcome(Outcome = ProofOutcome.Top, Message = "ensures unproven: Contract.Result<int>() * div >= n. The static checker determined that the condition '(((n - 1) / div + 1) * div) >= n' should hold on entry. Nevertheless, the condition may be too strong for the callers. If you think it is ok, add a precondition to document it: Contract.Requires((((n - 1) / div + 1) * div) >= n);", PrimaryILOffset = 66, MethodILOffset = 84)]*/
@Pure
	    private static @NonNegative int GetArrayLength(@NonNegative int n, @Positive int div)
	    {
	      // F: Made it a precondition
	      // Old:
	      // Contract.Assert(div > 0, "GetArrayLength: div arg must be greater than 0");
	      // New:
	      //Contract.Requires(div > 0);

	      // F: mine, to prove the postcondition
	      //Contract.Requires(n >= 0);

	      // F: I've added it
	      //Y: Contract.Ensures(Contract.Result<int>() >= 0);

	      // F: Hard to prove, as it involves a non-linear reasoning
	      //N: Contract.Ensures(Contract.Result<int>() <= n);

	      // F: I've added it, and now clousot can also check it
	      // F: Update, after an improvement in the polynomial expansion, we cannot prove it anymore
	      //N: Contract.Ensures(Contract.Result<int>() * div >= n);

	      return n > 0 ? (((n - 1) / div) + 1) : 0;
	    }

}
