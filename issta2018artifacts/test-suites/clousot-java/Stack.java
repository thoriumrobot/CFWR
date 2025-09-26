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

import org.checkerframework.checker.index.qual.GTENegativeOne;
import org.checkerframework.checker.index.qual.IndexOrHigh;
import org.checkerframework.checker.index.qual.LTEqLengthOf;
import org.checkerframework.checker.index.qual.LTLengthOf;
import org.checkerframework.common.value.qual.MinLen;

// Taken from the mscorlib.dll sources



// ==++==
// 
//   Copyright (c) Microsoft Corporation.  All rights reserved.
// 
// ==--==


/*=============================================================================
**
** Class: Stack
**
** <OWNER>kimhamil</OWNER>
**
** Purpose: An array implementation of a stack.
**
**
=============================================================================*/

  class System_Collections_Francesco_Stack
  {
    private Object @MinLen(1)[] _array;     // Storage for stack elements
    private @IndexOrHigh("_array") int  _size;           // Number of items in the stack.
    private int _version;        // Used to keep enumerator in sync w/ collection.
    
    private transient Object _syncRoot;

    private static final int _defaultCapacity = 10;

    /*[ContractInvariantMethod]
    private void ObjectInvariant()
    {*/
      // The invariant below is *just* commented for the regression test of numerical values
      //N: Contract.Invariant(_array != null);
      
      //Y: Contract.Invariant(_array.Length > 0);
      //Y: Contract.Invariant(_size >= 0);
      //Y: Contract.Invariant(_size <= _array.Length);
    /*}*/

    // The warning is ok: Cannot prove (of course...) that _array.Length > 0
    /*[ClousotRegressionTest]
    [RegressionOutcome(Outcome=ProofOutcome.Bottom,Message="invariant unreachable",PrimaryILOffset=12,MethodILOffset=9)]
    [RegressionOutcome(Outcome=ProofOutcome.Bottom,Message="invariant unreachable",PrimaryILOffset=30,MethodILOffset=9)]
    [RegressionOutcome(Outcome=ProofOutcome.Bottom,Message="invariant unreachable",PrimaryILOffset=55,MethodILOffset=9)]*/
    // We now detect early that because array == null, we never get to the invariant checks.
    //    [RegressionOutcome(Outcome = ProofOutcome.Top, Message = @"invariant unproven: _array.Length > 0", PrimaryILOffset = 12, MethodILOffset = 9)]
    //    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "invariant is valid", PrimaryILOffset = 30, MethodILOffset = 9)]
    //    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "invariant is valid", PrimaryILOffset = 55, MethodILOffset = 9)]
    public System_Collections_Francesco_Stack()
    {
    }

    // Create a stack with a specific initial capacity.  The initial capacity
    // must be a non-negative number.

    /*[ClousotRegressionTest]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "Array creation : ok", PrimaryILOffset = 63, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "ensures is valid", PrimaryILOffset = 22, MethodILOffset = 88)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "invariant is valid", PrimaryILOffset = 12, MethodILOffset = 88)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "invariant is valid", PrimaryILOffset = 30, MethodILOffset = 88)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "invariant is valid", PrimaryILOffset = 55, MethodILOffset = 88)]*/
    public System_Collections_Francesco_Stack(int initialCapacity)
    {
      //TODO: could use EnsuresLTLengthOf N: Contract.Ensures(this._array.Length >= initialCapacity);

      if (initialCapacity < 0)
        throw new RuntimeException();
      
      if (initialCapacity < _defaultCapacity)
        initialCapacity = _defaultCapacity;  // Simplify doubling logic in Push.
      
      _array = new Object[initialCapacity];
      _size = 0;
      _version = 0;
    }

       
    
      /*[ClousotRegressionTest]*/
    	public int get_Count()  { return _size; }

    
    
    // Removes all Objects from the Stack.
    /*[ClousotRegressionTest]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "invariant is valid", PrimaryILOffset = 12, MethodILOffset = 41)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "invariant is valid", PrimaryILOffset = 30, MethodILOffset = 41)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "invariant is valid", PrimaryILOffset = 55, MethodILOffset = 41)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "requires is valid", PrimaryILOffset = 13, MethodILOffset = 14)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "requires is valid", PrimaryILOffset = 31, MethodILOffset = 14)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "requires is valid", PrimaryILOffset = 64, MethodILOffset = 14)]*/
    public  void Clear()
    {
      /* Array.Clear(_array, 0, _size); // Don't need to doc this but we clear the elements so that the gc can reclaim the references. */
      _size = 0;
      _version++;
    }

    /*[ClousotRegressionTest]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "requires is valid", PrimaryILOffset = 13, MethodILOffset = 45)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "requires is valid", PrimaryILOffset = 31, MethodILOffset = 45)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "requires is valid", PrimaryILOffset = 55, MethodILOffset = 45)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "requires is valid", PrimaryILOffset = 79, MethodILOffset = 45)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "requires is valid", PrimaryILOffset = 98, MethodILOffset = 45)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "requires is valid", PrimaryILOffset = 132, MethodILOffset = 45)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "requires is valid", PrimaryILOffset = 166, MethodILOffset = 45)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = @"ensures is valid", PrimaryILOffset = 17, MethodILOffset = 68)]*/
    public  Object Clone()
    {
      // Contract.Ensures(Contract.Result<Object>() != null);

    	System_Collections_Francesco_Stack s = new System_Collections_Francesco_Stack(_size);

      s._size = _size;
      System.arraycopy(_array, 0, s._array, 0, _size);
      s._version = _version;
      
      return s;
    }


    // Wrong: should be able to prove it!
    /*[ClousotRegressionTest]
    [Pure]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "Lower bound access ok", PrimaryILOffset = 30, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "Upper bound access ok", PrimaryILOffset = 30, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "Lower bound access ok", PrimaryILOffset = 55, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "Upper bound access ok", PrimaryILOffset = 55, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "Lower bound access ok", PrimaryILOffset = 65, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "Upper bound access ok", PrimaryILOffset = 65, MethodILOffset = 0)]*/
    public  boolean Contains(Object obj)
    {
      @LTEqLengthOf("_array") int count = _size;

      while (count-- > 0)
      {
        if (obj == null)
        {
          if (_array[count] == null)
            return true;
        }
        else if (_array[count] != null && _array[count].equals(obj))
        {
          return true;
        }
      }
      return false;
    }

    // Copies the stack into an array.
    // main feature : array access objArray[i + index] using slack variables
    /*[ClousotRegressionTest]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "Lower bound access ok", PrimaryILOffset = 184, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "Upper bound access ok", PrimaryILOffset = 184, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "Lower bound access ok", PrimaryILOffset = 185, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "Upper bound access ok", PrimaryILOffset = 185, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "Lower bound access ok", PrimaryILOffset = 245, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "Upper bound access ok", PrimaryILOffset = 245, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "assert is valid", PrimaryILOffset = 126, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "assert is valid", PrimaryILOffset = 155, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "invariant is valid", PrimaryILOffset = 12, MethodILOffset = 274)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "invariant is valid", PrimaryILOffset = 30, MethodILOffset = 274)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "invariant is valid", PrimaryILOffset = 55, MethodILOffset = 274)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "assert is valid", PrimaryILOffset = 222, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "requires is valid", PrimaryILOffset = 19, MethodILOffset = 249)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "requires is valid", PrimaryILOffset = 43, MethodILOffset = 249)]*/
    public  void CopyTo(Object[] array, int index)
    {
      if (array == null)
        throw new IllegalArgumentException("array");
      /*if (array.Rank != 1)
        throw new Exception();*/
      if (index < 0)
        throw new RuntimeException();
      if (index + _size > array.length)
        throw new RuntimeException();
      // Contract.Assert(index + _size <= array.Length);

      int i = 0;
      if (array instanceof Object[])
      {
        //Y: Contract.Assert(index + _size <= array.length);
    	  @LTLengthOf(value = "array", offset="_size - 1") int _1 = index; 
    	  
        Object[] objArray = (Object[])array;
        //Y: Contract.Assert(index + _size <= objArray.length);
        @LTLengthOf(value = "objArray", offset="index - 1") int _2 = _size;
        while (i < _size)
        {
          objArray[i + index] = _array[_size - i - 1];
          i++;
        }
      }
      else
      {
        while (i < _size)
        {
          //Y: Contract.Assert(i + index < array.length);
        	@LTLengthOf(value = "array", offset="index") int _1 = i;

          array[ i + index] = (_array[_size - i - 1]);
          i++;
        }
      }
    }

    

    // Returns the top object on the stack without removing it.  If the stack
    // is empty, Peek throws an InvalidOperationException.
    /*[Pure]
    [ClousotRegressionTest]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "Lower bound access ok", PrimaryILOffset = 37, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "Upper bound access ok", PrimaryILOffset = 37, MethodILOffset = 0)]*/
    public Object Peek()
    {
      if (_size == 0)
        throw new RuntimeException();
      return _array[_size - 1];
    }

    // Pops an item from the top of the stack.  If the stack is empty, Pop
    // throws an InvalidOperationException.
    /*[ClousotRegressionTest]
    [RegressionOutcome (Outcome = ProofOutcome.True, Message = "Lower bound access ok", PrimaryILOffset = 60, MethodILOffset = 0)]
    [RegressionOutcome (Outcome = ProofOutcome.True, Message = "Upper bound access ok", PrimaryILOffset = 60, MethodILOffset = 0)]
    [RegressionOutcome (Outcome = ProofOutcome.True, Message = "Lower bound access ok", PrimaryILOffset = 75, MethodILOffset = 0)]
    [RegressionOutcome (Outcome = ProofOutcome.True, Message = "Upper bound access ok", PrimaryILOffset = 75, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "invariant is valid", PrimaryILOffset = 12, MethodILOffset = 81)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "invariant is valid", PrimaryILOffset = 30, MethodILOffset = 81)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "invariant is valid", PrimaryILOffset = 55, MethodILOffset = 81)]*/
    public Object Pop()
    {
      if (_size == 0)
        throw new RuntimeException();
      _version++;
      Object obj = _array[--_size];
      _array[_size] = null;     // Free memory quicker.
      return obj;
    }

    // Wrong: should be able to prove it!

    // Pushes an item to the top of the stack.
    // F: Main feature : array creation with multiplication
    /*[ClousotRegressionTest]
    [RegressionOutcome (Outcome = ProofOutcome.True, Message = "Array creation : ok", PrimaryILOffset = 35, MethodILOffset = 0)]
    [RegressionOutcome (Outcome = ProofOutcome.True, Message = "Lower bound access ok", PrimaryILOffset = 116, MethodILOffset = 0)]
    [RegressionOutcome (Outcome = ProofOutcome.True, Message = "Upper bound access ok", PrimaryILOffset = 116, MethodILOffset = 0)]
    [RegressionOutcome (Outcome = ProofOutcome.True, Message = "assert is valid", PrimaryILOffset = 85, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "invariant is valid", PrimaryILOffset = 12, MethodILOffset = 131)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "invariant is valid", PrimaryILOffset = 30, MethodILOffset = 131)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "invariant is valid", PrimaryILOffset = 55, MethodILOffset = 131)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "requires is valid", PrimaryILOffset = 13, MethodILOffset = 56)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "requires is valid", PrimaryILOffset = 31, MethodILOffset = 56)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "requires is valid", PrimaryILOffset = 55, MethodILOffset = 56)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "requires is valid", PrimaryILOffset = 79, MethodILOffset = 56)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "requires is valid", PrimaryILOffset = 98, MethodILOffset = 56)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "requires is valid", PrimaryILOffset = 132, MethodILOffset = 56)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "requires is valid", PrimaryILOffset = 166, MethodILOffset = 56)]*/
    public void Push(Object obj)
    {
      if (_size == _array.length)
      {
        Object @MinLen(1)[] newArray = new Object[2 * _array.length];
        System.arraycopy(_array, 0, newArray, 0, _size);

        _array = newArray;
        //Y: Contract.Assert(_size < _array.Length);
        @LTLengthOf("_array") int _1 = _size;
      }
      //{
      //  Contract.Assert(_size < _array.Length);
      //}
      _array[_size++] = obj;
      _version++;
    }


    // Copies the Stack to an array, in the same order Pop would return the items.
    /*[ClousotRegressionTest] // main feature : array access with two variables
    [RegressionOutcome (Outcome = ProofOutcome.True, Message = "Array creation : ok", PrimaryILOffset = 7, MethodILOffset = 0)]
    [RegressionOutcome (Outcome = ProofOutcome.True, Message = "Lower bound access ok", PrimaryILOffset = 36, MethodILOffset = 0)]
    [RegressionOutcome (Outcome = ProofOutcome.True, Message = "Upper bound access ok", PrimaryILOffset = 36, MethodILOffset = 0)]
    [RegressionOutcome (Outcome = ProofOutcome.True, Message = "Lower bound access ok", PrimaryILOffset = 37, MethodILOffset = 0)]
    [RegressionOutcome (Outcome = ProofOutcome.True, Message = "Upper bound access ok", PrimaryILOffset = 37, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "invariant is valid", PrimaryILOffset = 12, MethodILOffset = 61)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "invariant is valid", PrimaryILOffset = 30, MethodILOffset = 61)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "invariant is valid", PrimaryILOffset = 55, MethodILOffset = 61)]*/
    public Object[] ToArray()
    {
      Object[] objArray = new Object[_size];
      int i = 0;
      while (i < _size)
      {
        objArray[i] = _array[_size - i - 1];
        i++;
      }
      return objArray;
    }

    private class StackEnumerator 
    {
      private System_Collections_Francesco_Stack _stack;
      private @LTLengthOf("_stack._array") int _index;
      private int _version;
      private Object currentElement;

      /*[ContractInvariantMethod]
      private void ObjectInvariant()
      {*/
        // Contract.Invariant(_stack != null);
        // Contract.Invariant(_stack._array != null);

        //Y: Contract.Invariant(_index < _stack._array.Length);
        //N: Contract.Invariant(_stack._size <= _stack._array.Length);
      /*}*/

      /*[ClousotRegressionTest]
      [RegressionOutcome(Outcome = ProofOutcome.True, Message = "invariant is valid", PrimaryILOffset = 22, MethodILOffset = 73)]
      [RegressionOutcome(Outcome = ProofOutcome.True, Message = "invariant is valid", PrimaryILOffset = 57, MethodILOffset = 73)]*/
      StackEnumerator(System_Collections_Francesco_Stack stack)
      {
        // Contract.Requires(stack != null);
        // Contract.Requires(stack._array != null);
        //TODO: maybe could use precondition N: Contract.Requires(stack._size <= stack._array.Length);

        _stack = stack;
        _version = _stack._version;
        _index = -2;
        currentElement = null;
      }

      

      /*[ClousotRegressionTest] // main feature : a difficult invariant proof
      [RegressionOutcome(Outcome = ProofOutcome.True, Message = "Lower bound access ok", PrimaryILOffset = 162, MethodILOffset = 0)]
      [RegressionOutcome(Outcome = ProofOutcome.True, Message = "Upper bound access ok", PrimaryILOffset = 162, MethodILOffset = 0)]
      [RegressionOutcome(Outcome = ProofOutcome.True, Message = "Lower bound access ok", PrimaryILOffset = 301, MethodILOffset = 0)]
      [RegressionOutcome(Outcome = ProofOutcome.True, Message = "Upper bound access ok", PrimaryILOffset = 301, MethodILOffset = 0)]
      [RegressionOutcome(Outcome = ProofOutcome.True, Message = "assert is valid", PrimaryILOffset = 22, MethodILOffset = 0)]
      [RegressionOutcome(Outcome = ProofOutcome.True, Message = "assert is valid", PrimaryILOffset = 73, MethodILOffset = 0)]
      [RegressionOutcome(Outcome = ProofOutcome.True, Message = "assert is valid", PrimaryILOffset = (int)190, MethodILOffset = 0)]
      [RegressionOutcome(Outcome = ProofOutcome.True, Message = "assert is valid", PrimaryILOffset = 241, MethodILOffset = 0)]
      [RegressionOutcome(Outcome = ProofOutcome.True, Message = "assert is valid", PrimaryILOffset = 337, MethodILOffset = 0)]
      [RegressionOutcome(Outcome = ProofOutcome.True, Message = "invariant is valid", PrimaryILOffset = 22, MethodILOffset = 348)]
      [RegressionOutcome(Outcome = ProofOutcome.True, Message = "invariant is valid", PrimaryILOffset = 57, MethodILOffset = 348)]*/
      public boolean MoveNext()
      {
        //Y: Contract.Assert(_index < _stack._array.Length);
        @LTLengthOf("_stack._array") int _1 = _index;
        
        boolean retval;
        if (_version != _stack._version)
        {
          //Y: Contract.Assert(_index < _stack._array.Length);
        	@LTLengthOf("_stack._array") int _2 = _index;

          throw new RuntimeException();
        }

        if (_index == -2)
        {  // First call to enumerator.
          _index = _stack._size - 1;
          retval = (_index >= 0);

          if (retval)
          {
            currentElement = _stack._array[_index];
          }

          //Y: Contract.Assert(_index < _stack._array.Length);
          @LTLengthOf("_stack._array") int _3 = _index;
          
          return retval;
        }
        if (_index == -1)
        {  // End of enumeration.
          
          //Y: Contract.Assert(_index < _stack._array.Length);
        	@LTLengthOf("_stack._array") int _4 = _index;
          
          return false;
        }

        retval = (--_index >= 0);
        if (retval)
          currentElement = _stack._array[_index];
        else
          currentElement = null;
        
        //Y: Contract.Assert(_index < _stack._array.Length);
        @LTLengthOf("_stack._array") int _5 = _index;
      
        return retval;
      }

      
      
              /*[ClousotRegressionTest]*/
        		public Object  get_Current()
        {
          if (_index == -2)
            throw new RuntimeException();

          if (_index == -1)
            throw new RuntimeException();

          return currentElement;
        }
      

      /*[ClousotRegressionTest]
      [RegressionOutcome(Outcome = ProofOutcome.True, Message = "invariant is valid", PrimaryILOffset = 22, MethodILOffset = 45)]
      [RegressionOutcome(Outcome = ProofOutcome.True, Message = "invariant is valid", PrimaryILOffset = 57, MethodILOffset = 45)]*/
      public void Reset()
      {
        if (_version != _stack._version)
          throw new RuntimeException();

        _index = -2;
        currentElement = null;
      }
    }

  }



