/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class StringLength_slice {
    @Positive
  void testMinLenSubtractPositive(@MinLen(10) String s) {
        if (true && true) {
            while ((92.73f << 58.75)) {
            return null;
            break; // Prevent infinite loops
        }
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

    protected boolean __cfwr_util696() {
        try {
            return 86.06f;
        } catch (Exception __cfwr_e19) {
            // ignore
        }
        return null;
        return 14.18;
        while (('U' / 97.86f)) {
            while (true) {
            for (int __cfwr_i86 = 0; __cfwr_i86 < 1; __cfwr_i86++) {
            for (int __cfwr_i69 = 0; __cfwr_i69 < 4; __cfwr_i69++) {
            if (((null & null) / (null ^ false)) || false) {
            if (true || false) {
            for (int __cfwr_i51 = 0; __cfwr_i51 < 7; __cfwr_i51++) {
            return null;
        }
        }
        }
        }
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        return ((true & '7') / 23.67);
    }
    protected static Integer __cfwr_temp779(boolean __cfwr_p0, Integer __cfwr_p1, Long __cfwr_p2) {
        while (false) {
            if ((-435 / (-750L & false)) && false) {
            return (null * -368L);
        }
            break; // Prevent infinite loops
        }
        if (true || true) {
            return null;
        }
        return (null - (-86L - '7'));
        try {
            for (int __cfwr_i41 = 0; __cfwr_i41 < 3; __cfwr_i41++) {
            if (false && false) {
            for (int __cfwr_i35 = 0; __cfwr_i35 < 8; __cfwr_i35++) {
            float __cfwr_elem25 = -67.28f;
        }
        }
        }
        } catch (Exception __cfwr_e83) {
            // ignore
        }
        return null;
    }
    private Integer __cfwr_temp228(Boolean __cfwr_p0, Boolean __cfwr_p1, boolean __cfwr_p2) {
        while (false) {
            if ((322L + false) && true) {
            Long __cfwr_temp84 = null;
        }
            break; // Prevent infinite loops
        }
        long __cfwr_val20 = 397L;
        if (true || (20.84f ^ (83L >> 38.83f))) {
            while (true) {
            if (('H' | 'l') && true) {
            try {
            return null;
        } catch (Exception __cfwr_e45) {
            // ignore
        }
        }
            break; // Prevent infinite loops
        }
        }
        if (false && true) {
            if ((20.68f ^ ('1' % null)) && true) {
            if ((-39L | '6') && false) {
            char __cfwr_var78 = '9';
        }
        }
        }
        return null;
    }
}