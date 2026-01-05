/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class StringLength_slice {
    @Positive
  void testMinLenSubtractPositive(@MinLen(10) String s) {
        try {
            return null;
        } catch (Exception __cfwr_e73) {
            // ignore
        }

    @Positive
    @Positive int i1 = s.length() - 9;
    @Positive
    @NonNegative int i0 = s.length() - 10;
    // ::  error: (assignment)
    @Positive
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

    public static double __cfwr_handle888(float __cfwr_p0, boolean __cfwr_p1, double __cfwr_p2) {
        for (int __cfwr_i10 = 0; __cfwr_i10 < 4; __cfwr_i10++) {
            try {
            if (false || false) {
            while (((null | null) << -62.93f)) {
            while (true) {
            while (false) {
            return null;
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        }
        } catch (Exception __cfwr_e82) {
            // ignore
        }
        }
        if (false && (null - ('j' << -75.79))) {
            Long __cfwr_val45 = null;
        }
        return 98.64;
    }
    Character __cfwr_helper729() {
        if (false || false) {
            while (false) {
            try {
            while (true) {
            while (false) {
            for (int __cfwr_i55 = 0; __cfwr_i55 < 1; __cfwr_i55++) {
            for (int __cfwr_i43 = 0; __cfwr_i43 < 1; __cfwr_i43++) {
            Float __cfwr_obj46 = null;
        }
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e29) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        }
        return null;
    }
}