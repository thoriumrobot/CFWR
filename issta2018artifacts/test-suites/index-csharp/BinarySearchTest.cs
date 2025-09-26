using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;

public class BinarySearchTest {

    [ContractInvariantMethod]
    private void Invariant()
    {
        Contract.Invariant(iTransitions.Length == iNameKeys.Length);
        Contract.Invariant(iNameKeys.Length == iTransitions.Length);
    }

    private readonly long [] iTransitions;
    private readonly String [] iNameKeys;

    private BinarySearchTest(
            long [] transitions,
            String [] nameKeys) {
        Contract.Assume(transitions.Length == iNameKeys.Length);
        Contract.Assume(nameKeys.Length == iTransitions.Length);
        iTransitions = transitions;
        iNameKeys = nameKeys;
    }

    public String getNameKey(long instant) {
        long[] transitions = iTransitions;
        int i = Array.BinarySearch(transitions, instant);
        if (i >= 0) {
            return iNameKeys[i];
        }
        i = ~i;
        if (i > 0) {
            return iNameKeys[i - 1];
        }
        return "";
    }

    public String getNameKey2(long instant) {
        long[] transitions = iTransitions;
        int i = Array.BinarySearch(transitions, instant);
        if (i >= 0) {
            return iNameKeys[i];
        }
        i = ~i;
        if (i < iNameKeys.Length) {
            return iNameKeys[i];
        }
        return "";
    }
}
