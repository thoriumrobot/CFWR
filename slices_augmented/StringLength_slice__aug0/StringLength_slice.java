/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class StringLength_slice {
    @Positive
  void testMinLenSubtractPositive(@MinLen(10) String s) {
        for (int __cfwr_i28 = 0; __cfwr_i28 < 7; __cfwr_i28++) {
            return false;
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

    private short __cfwr_compute548(double __cfwr_p0) {
        try {
            if (false && false) {
            for (int __cfwr_i40 = 0; __cfwr_i40 < 1; __cfwr_i40++) {
            return -72.38f;
        }
        }
        } catch (Exception __cfwr_e93) {
            // ignore
        }
        while ((7.33 + 67.74f)) {
            if (((null % true) * (47.14f & -739)) && true) {
            for (int __cfwr_i92 = 0; __cfwr_i92 < 3; __cfwr_i92++) {
            if (true || true) {
            return (-145 / -90.59);
        }
        }
        }
            break; // Prevent infinite loops
        }
        for (int __cfwr_i51 = 0; __cfwr_i51 < 4; __cfwr_i51++) {
            long __cfwr_temp92 = -528L;
        }
        while (false) {
            while (true) {
            try {
            try {
            if ((-817 - 76.69) && false) {
            try {
            char __cfwr_entry96 = 'n';
        } catch (Exception __cfwr_e16) {
            // ignore
        }
        }
        } catch (Exception __cfwr_e4) {
            // ignore
        }
        } catch (Exception __cfwr_e34) {
            // ignore
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        return (false << (-141L >> 0.22f));
    }
    protected Long __cfwr_calc862(Boolean __cfwr_p0, long __cfwr_p1, Double __cfwr_p2) {
        return null;
        Double __cfwr_temp47 = null;
        return null;
    }
    protected long __cfwr_compute989(Double __cfwr_p0, byte __cfwr_p1, Object __cfwr_p2) {
        return null;
        for (int __cfwr_i2 = 0; __cfwr_i2 < 5; __cfwr_i2++) {
            return null;
        }
        return (null * -82.31);
    }
}