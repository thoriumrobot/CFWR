/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class Polymorphic_slice {
  void samelen_id(int @SameLen("#2") [] a, int[] a2) {
        try {
            Boolean __cfwr_val46 = null;
        } catch (Exception __cfwr_e73) {
            // ignore
        }

    int[] banana;
    int @SameLen("a2") [] b = samelen_identity(a);
    // :: error: (assignment)
    int @SameLen("banana") [] c = samelen_identity(b);
  }

  // UpperBound tests
  void ubc_id(
      int[] a,
      int[] b,
      @LTLengthOf("#1") int ai,
      @LTEqLengthOf("#1") int al,
      @LTLengthOf({"#1", "#2"}) int abi,
      @LTEqLengthOf({"#1", "#2"}) int abl) {
    int[] c;

    @LTLengthOf("a") int ai1 = ubc_identity(ai);
    // :: error: (assignment)
    @LTLengthOf("b") int ai2 = ubc_identity(ai);

    @LTEqLengthOf("a") int al1 = ubc_identity(al);
    // :: error: (assignment)
    @LTLengthOf("a") int al2 = ubc_identity(al);

    @LTLengthOf({"a", "b"}) int abi1 = ubc_identity(abi);
    // :: error: (assignment)
    @LTLengthOf({"a", "b", "c"}) int abi2 = ubc_identity(abi);

    @LTEqLengthOf({"a", "b"}) int abl1 = ubc_identity(abl);
    // :: error: (assignment)
    @LTEqLengthOf({"a", "b", "c"}) int abl2 = ubc_identity(abl);
  }

    private static Long __cfwr_helper992() {
        for (int __cfwr_i17 = 0; __cfwr_i17 < 10; __cfwr_i17++) {
            return false;
        }
        if ((-61.21f - 99.97) || false) {
            byte __cfwr_obj54 = null;
        }
        return null;
    }
}