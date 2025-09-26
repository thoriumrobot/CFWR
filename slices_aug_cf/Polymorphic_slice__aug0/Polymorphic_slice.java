/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class Polymorphic_slice {
  void samelen_id(int @SameLen("#2") [] a, int[] a2) {
        try {
            return null;
        } catch (Exception __cfwr_e2) {
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

    protected float __cfwr_handle187() {
        return null;
        return -53.89f;
    }
    public static boolean __cfwr_compute341() {
        byte __cfwr_result6 = null;
        if (((null * 56.63f) >> 'v') && false) {
            for (int __cfwr_i27 = 0; __cfwr_i27 < 1; __cfwr_i27++) {
            Double __cfwr_entry17 = null;
        }
        }
        int __cfwr_temp35 = (-77 * (95.04 - -13.60));
        return false;
    }
}