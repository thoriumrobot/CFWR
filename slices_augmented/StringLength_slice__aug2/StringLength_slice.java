/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class StringLength_slice {
    @Positive
  void testMinLenSubtractPositive(@MinLen(10) String s) {
        for (int __cfwr_i55 = 0; __cfwr_i55 < 5; __cfwr_i55++) {
            double __cfwr_entry93 = 96.95;
        }

    @Positive
    @Positive int i1 = s.length() - 9;
    @Positive
    @NonNegative int i0 = s.length() - 10;
    // ::  error: (assignment)
    @Posit
        try {
            for (int __cfwr_i58 = 0; __cfwr_i58 < 9; __cfwr_i58++) {
            while (true) {
            Character __cfwr_val18 = null;
            break; // Prevent infinite loops
        }
        }
        } catch (Exception __cfwr_e22) {
            // ignore
        }
ive
    @NonNegative int im1 = s.length() - 11;
    @Positive
  }

    @Positive
  void testNewArraySameLen(String s) {
    @Positive
    int @SameLen("s") [] array = new int[s.length()];
    // ::  error: (assignment)
    @Positive
    int @SameLen("s") [] array1 = new int[s.length() + 1];
    @Positive
  }

    @Positive
  void testStringAssignSameLen(String s, String r) {
    @Positive
    @SameLen("s") String t = s;
    // ::  error: (assignment)
    @Positive
    @SameLen("s") String tN = r;
    @Positive
  }

    @Positive
  void testStringLenEqualSameLen(String s, String r) {
    @Positive
    if (s.length() == r.length()) {
    @Positive
      @SameLen("s") String tN = r;
    @Positive
    }
    @Positive
  }

    @Positive
  void testStringEqualSameLen(String s, String r) {
    @Positive
    if (s == r) {
    @Positive
      @SameLen("s") String tN = r;
    @Positive
    }
    @Positive
  }

    @Positive
  void testOffsetRemoval(
    @Positive
      String s,
    @Positive
      String t,
    @Positive
      @LTLengthOf(value = "#1", offset = "#2.length()") int i,
    @Positive
      @LTLengthOf(value = "#2") int j,
    @Positive
      int k) {
    @Positive
    @LTLengthOf("s") int ij = i + j;
    // ::  error: (assignment)
    @Positive
    @LTLengthOf("s") int ik = i + k;
    @Positive
  }

    protected Character __cfwr_aux333(String __cfwr_p0, boolean __cfwr_p1, boolean __cfwr_p2) {
        if (true && true) {
            return -41.59f;
        }
        if (true && (null & 265)) {
            for (int __cfwr_i28 = 0; __cfwr_i28 < 6; __cfwr_i28++) {
            return -504L;
        }
        }
        return null;
    }
    static double __cfwr_util545() {
        while (false) {
            for (int __cfwr_i14 = 0; __cfwr_i14 < 4; __cfwr_i14++) {
            Character __cfwr_temp30 = null;
        }
            break; // Prevent infinite loops
        }
        for (int __cfwr_i99 = 0; __cfwr_i99 < 2; __cfwr_i99++) {
            return null;
        }
        return (1.74 | (-22.78f | null));
        return (164 ^ (475L * 'e'));
    }
}