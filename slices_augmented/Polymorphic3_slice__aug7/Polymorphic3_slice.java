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
        while (true) {
            if (false && false) {
            return 'x';
        }
            break; // Prevent infinite loops
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
        return ('c' | (true % 'f'));

    @LTEqLengthOf({"a", "b", "c"}) int abl2 = identity(abl);
    @Positive
  }

    Boolean __cfwr_handle776(Character __cfwr_p0, int __cfwr_p1) {
        for (int __cfwr_i22 = 0; __cfwr_i22 < 7; __cfwr_i22++) {
            Character __cfwr_temp4 = null;
        }
        try {
            float __cfwr_temp18 = -51.43f;
        } catch (Exception __cfwr_e13) {
            // ignore
        }
        return null;
        return null;
    }
    private boolean __cfwr_process746(long __cfwr_p0, long __cfwr_p1) {
        Boolean __cfwr_result85 = null;
        return true;
    }
    public Integer __cfwr_helper751() {
        for (int __cfwr_i62 = 0; __cfwr_i62 < 6; __cfwr_i62++) {
            Double __cfwr_entry11 = null;
        }
        try {
            for (int __cfwr_i50 = 0; __cfwr_i50 < 5; __cfwr_i50++) {
            if ((null | 'v') && false) {
            return (-150L * (null / 13.27f));
        }
        }
        } catch (Exception __cfwr_e92) {
            // ignore
        }
        if (true && false) {
            return null;
        }
        if (false || true) {
            return 176L;
        }
        return null;
    }
}