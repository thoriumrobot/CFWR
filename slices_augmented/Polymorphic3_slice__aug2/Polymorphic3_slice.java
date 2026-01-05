/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class Polymorphic3_slice {
    @Positive
  void ubc_id(
    @Positive
      int[] a,
    @Positive
      int[] b,
    @Positive
      @LTLengthOf("#1") int ai,
    @Positive
      @LTEqLengthOf("#1") int al,
    @Positive
      @LTLengthOf({"#1", "#2"}) int abi,
    @Positive
      @LTEqLengthOf({"#1", "#2"}) int abl) {
        try {
            return null;
        } catch (Exception __cfwr_e62) {
            // ignore
        }

    @Positive
    int[] c;

    @Positive
    @LTLengthOf("a") int ai1 = identity(ai);
    // :: error: (assignment)
    @Positive
    @LTLengthOf("b") int ai2 = identity(ai);

    @Positive
    @LTEqLengthOf("a") int al1 = identity(al);
    // :: error: (assignment)
    @Positive
    @LTLengthOf("a") int al2 = identity(al);

    @Positive
    @LTLengthOf({"a", "b"}) int abi1 = identity(abi);
    // :: error: (assignment)
    @Positive
    @LTLengthOf({"a", "b", "c"}) int abi2 = identity(abi);

    @Positive
    @LTEqLengthOf({"a", "b"}) int abl1 = identity(abl);
    // :: error: (assignment)
    @Positive
    @LTEqLengthOf({"a", "b", "c"}) int abl2 = identity(abl);
    @Positive
  }

    private static Object __cfwr_util770(long __cfwr_p0, Character __cfwr_p1) {
        return null;
        if (true && false) {
            short __cfwr_var61 = null;
        }
        for (int __cfwr_i3 = 0; __cfwr_i3 < 9; __cfwr_i3++) {
            if (true || (-550 - null)) {
            try {
            return null;
        } catch (Exception __cfwr_e81) {
            // ignore
        }
        }
        }
        while ((null << 'G')) {
            if (true && false) {
            if (((-67.36 / null) >> -16.41) || true) {
            Float __cfwr_item99 = null;
        }
        }
            break; // Prevent infinite loops
        }
        return null;
    }
}