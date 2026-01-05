/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class StringLength_slice {
    @Positive
  void testMinLenSubtractPositive(@MinLen(10) String s) {
        return null;

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
  void testNewAr
        for (int __cfwr_i95 = 0; __cfwr_i95 < 2; __cfwr_i95++) {
            while ((49.69 * (null << -42.15))) {
            if ((null * (-99.70f | -60.90f)) || (62 ^ false)) {
            String __cfwr_elem3 = "hello48";
        }
            break; // Prevent infinite loops
        }
        }
raySameLen(String s) {
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

    public long __cfwr_temp234(long __cfwr_p0, char __cfwr_p1) {
        try {
            try {
            for (int __cfwr_i55 = 0; __cfwr_i55 < 7; __cfwr_i55++) {
            for (int __cfwr_i24 = 0; __cfwr_i24 < 8; __cfwr_i24++) {
            for (int __cfwr_i94 = 0; __cfwr_i94 < 3; __cfwr_i94++) {
            if (((98.30 + '4') << (-6.31f + 38.45f)) || false) {
            Boolean __cfwr_var19 = null;
        }
        }
        }
        }
        } catch (Exception __cfwr_e79) {
            // ignore
        }
        } catch (Exception __cfwr_e79) {
            // ignore
        }
        for (int __cfwr_i97 = 0; __cfwr_i97 < 6; __cfwr_i97++) {
            if (true && ((-65.87f * false) & 947)) {
            if (((-667 >> 355L) << null) || (479 & null)) {
            for (int __cfwr_i85 = 0; __cfwr_i85 < 9; __cfwr_i85++) {
            while (true) {
            while (false) {
            for (int __cfwr_i34 = 0; __cfwr_i34 < 7; __cfwr_i34++) {
            char __cfwr_result7 = 'O';
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        }
        }
        }
        }
        return 99L;
    }
    static Character __cfwr_temp348() {
        if (true || false) {
            return "test32";
        }
        while ((null / (-63L - -915))) {
            if (false && true) {
            while (false) {
            for (int __cfwr_i89 = 0; __cfwr_i89 < 1; __cfwr_i89++) {
            return null;
        }
            break; // Prevent infinite loops
        }
        }
            break; // Prevent infinite loops
        }
        return null;
    }
}