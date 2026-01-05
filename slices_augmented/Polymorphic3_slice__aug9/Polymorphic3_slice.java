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
        return null;

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

    protected Character __cfwr_compute783() {
        try {
            Float __cfwr_item82 = null;
        } catch (Exception __cfwr_e83) {
            // ignore
        }
        while (true) {
            return null;
            break; // Prevent infinite loops
        }
        return null;
    }
    static byte __cfwr_temp416(long __cfwr_p0, Object __cfwr_p1, Long __cfwr_p2) {
        for (int __cfwr_i94 = 0; __cfwr_i94 < 5; __cfwr_i94++) {
            if (true || false) {
            while ((null << 107)) {
            try {
            if (false || false) {
            for (int __cfwr_i89 = 0; __cfwr_i89 < 5; __cfwr_i89++) {
            byte __cfwr_data94 = null;
        }
        }
        } catch (Exception __cfwr_e27) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        }
        }
        return null;
        byte __cfwr_var27 = (false ^ -81.07);
        return null;
    }
}