using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;

public class MinLenFieldInvar {
    class Super {
        public readonly int [] minlen2;

        [ContractInvariantMethod]
        private void Invariant()
        {
            Contract.Invariant(minlen2.Length >= 2);
        }

        public Super(int [] minlen2) {
            Contract.Requires(minlen2.Length >= 2);
            this.minlen2 = minlen2;
        }
    }

    // :: error: (field.invariant.not.subtype)
    class InvalidSub : Super {

        [ContractInvariantMethod]
        private void Invariant()
        {
            Contract.Invariant(minlen2.Length >= 1);
        }


        public InvalidSub() : base(new int[] { 1, 2 })
        {
            ;
        }
    }

    class ValidSub : Super {
        public readonly int[] validSubField;

        [ContractInvariantMethod]
        private void Invariant()
        {
            Contract.Invariant(minlen2.Length >= 4);
        }

        public ValidSub(int[] validSubField) : base(new int[] { 1, 2, 3, 4 })
        {
            ;
            this.validSubField = validSubField;
        }
    }

    // :: error: (field.invariant.not.found.superclass)
    class InvalidSubSub1 : ValidSub {

        [ContractInvariantMethod]
        private void Invariant()
        {
            Contract.Invariant(validSubField.Length >= 3);
        }

        public InvalidSubSub1():
            base(new int[] {1, 2}){
        }
    }

    // :: error: (field.invariant.not.subtype.superclass)
    class InvalidSubSub2 : ValidSub {

        [ContractInvariantMethod]
        private void Invariant()
        {
            Contract.Invariant(minlen2.Length >= 3);
        }

        public InvalidSubSub2() : base(new int[] { 1, 2 })
        {
            ;
        }
    }
#if BOTTOM
    @FieldInvariant(field = "minlen2", qualifier = BottomVal.class)
    @MinLenFieldInvariant(field = "validSubField", minLen = 4)
    class ValidSubSub : ValidSub {
        public ValidSubSub() {
            base(null);
        }

        void test() {

            int @BottomVal [] bot = minlen2;

            int[] four = validSubField;
            if(TestHelper.nondet()) Contract.Assert(four.Length >= 4);
        }
    }
#endif
}
