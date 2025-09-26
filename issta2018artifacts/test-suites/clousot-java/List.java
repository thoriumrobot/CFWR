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

import java.util.Comparator;

import org.checkerframework.checker.index.qual.IndexOrHigh;
import org.checkerframework.checker.index.qual.LTEqLengthOf;
import org.checkerframework.checker.index.qual.LTLengthOf;
import org.checkerframework.checker.index.qual.Positive;


// F: We have no regression attribute for that. 
// In fact now Closuot has problems with generics!



// ==++==
// 
//   Copyright (c) Microsoft Corporation.  All rights reserved.
// 
// ==--==
/*============================================================
**
** Class:  List
** 
** <OWNER>kimhamil</OWNER>
**
** Purpose: Implements a generic, dynamically sized list as an 
**          array.
**
** 
===========================================================*/

  // Implements a variable-size List that uses an array of objects to store the
  // elements. A List has a capacity, which is the allocated length
  // of the internal array. As elements are added to a List, the capacity
  // of the List is automatically increased as required by reallocating the
  // internal array.
  // 
  //[DebuggerTypeProxy(typeof(Mscorlib_CollectionDebugView<>))]

  public class List<T> 
  {
    private static final int _defaultCapacity = 4;

    private T[] _items;
    //N: [ContractPublicPropertyName("Count")]
    private @IndexOrHigh("_items") int _size;
    private int _version;
    
    private transient Object _syncRoot;


    /*[ContractInvariantMethod]
    private void ObjectInvariant()
    {*/
      //N: Contract.Invariant(_emptyArray != null);
      //N: Contract.Invariant(_items != null);
      //Y: Contract.Invariant(_size >= 0);
      //Y: Contract.Invariant(_size <= _items.Length);
    /*}*/
    
    // Read-only property describing how many elements are in the List.
    public @IndexOrHigh("_items") int get_Count()
    {
       
        //NY: Contract.Ensures(Contract.Result<int>() == _size);

        return _size;
      
    }
    
    private void EnsureCapacity(int min)
    {
      //N: Contract.Ensures(this._items.Length >= min);
      //N: Contract.Ensures(this._size == Contract.OldValue<int>(this._size));
      if (_items.length < min)
      {
        int newCapacity = _items.length == 0 ? _defaultCapacity : _items.length * 2;
        //Contract.Assert(newCapacity >= _items.Length);
        
        if (newCapacity < min) 
          newCapacity = min;
        
        set_Capacity(newCapacity);
        
        //Y: Contract.Assert(_items.Length >= min);
        @LTEqLengthOf("_items") int _1 = min;
      }
    }
  
    

    // Adds the given object to the end of this list. The size of the list is
    // increased by one. If required, the capacity of the list is doubled
    // before adding the new element.
    //
    /*[ClousotRegressionTest]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message="Lower bound access ok",PrimaryILOffset=92,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message="Upper bound access ok",PrimaryILOffset=92,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message="assert is valid",PrimaryILOffset=61,MethodILOffset=0)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message="invariant is valid",PrimaryILOffset=12,MethodILOffset=111)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message="invariant is valid",PrimaryILOffset=30,MethodILOffset=111)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message="invariant is valid",PrimaryILOffset=48,MethodILOffset=111)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message="invariant is valid",PrimaryILOffset=73,MethodILOffset=111)]
    [RegressionOutcome(Outcome=ProofOutcome.True,Message="ensures is valid",PrimaryILOffset=28,MethodILOffset=111)]*/
    public void Add(T item)
    {
      if (_size == _items.length) 
      { 
        EnsureCapacity(_size + 1); 
        
        //Y: Contract.Assert(_items.length >= _size + 1);
        @LTEqLengthOf("_items") int _1 = _size + 1;
      }
      
      _items[_size++] = item;
      _version++;
    }

    

    // Contains returns true if the specified element is in the List.
    // It does a linear, O(n) search.  Equality is determined by calling
    // item.Equals().
    //
    /*[ClousotRegressionTest]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "Lower bound access ok", PrimaryILOffset = 110, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "Upper bound access ok", PrimaryILOffset = 110, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "assert is valid", PrimaryILOffset = 59, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "assert is valid", PrimaryILOffset = 138, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "ensures is valid", PrimaryILOffset = 31, MethodILOffset = 171)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "Lower bound access ok", PrimaryILOffset = 29, MethodILOffset = 0)]
    [RegressionOutcome(Outcome = ProofOutcome.True, Message = "Upper bound access ok", PrimaryILOffset = 29, MethodILOffset = 0)]*/
    public boolean Contains(T item)
    {
      if ((Object)item == null)
      {
        for (int i = 0; i < _size; i++)
          if ((Object)_items[i] == null)
          {
        	  
            //Y: Contract.Assert(Count > 0);
        	  @Positive int _1 = get_Count();
            return true;
          }
        return false;
      }
      else
      {
    	
        System_Collections_Generic_EqualityComparer<T> c = System_Collections_Generic_EqualityComparer.Default();
        for (int i = 0; i < _size; i++)
        {
          if (c.Equals(_items[i], item))
          {
            //Y: Contract.Assert(Count > 0);
        	  @Positive int _1 = get_Count();
            return true;
          }
        }
        return false;
      }
    }
    
    private static class System_Collections_Generic_EqualityComparer<T>{
    	public static <T>System_Collections_Generic_EqualityComparer<T> Default() {return null;}
    	public boolean Equals(T a, T b) {return false;}
    }
    
    
    public void set_Capacity(int value)
    {
     
     
      
        //N: Contract.Ensures(_items.Length <= value);
        //TODO could be by EnsuresLTLengthOf N: Contract.Ensures(_items.Length >= value);
        //N: Contract.Ensures(_size == Contract.OldValue<int>(_size));
        if (value < _size)
        {
          throw new RuntimeException();
        }

        if (value != _items.length)
        {
          if (value > 0)
          {
            T[] newItems = (T[])new Object[value];
            if (_size > 0)
            {
              System.arraycopy(_items, 0, newItems, 0, _size);
            }
            _items = newItems;
          }
          else
          {
            _items = (T[])new Object[0];
          }
        }
      }
    
    
  }

 
