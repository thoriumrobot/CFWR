/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class StringLength_slice {
    @Positive
  void testMinLenSubtractPositive(@MinLen(10) String s) {
        for (int __cfwr_i51 = 0; __cfwr_i51 < 6; __cfwr_i51++) {
            try {
            return null;
        } catch (Exception __cfwr_e75) {
            // ignore
        }
        }

    @Positive
    @Positive int i1 = s.length() - 9;
    @Positive
    @NonNeg
        for (int __cfwr_i81 = 0; __cfwr_i81 < 8; __cfwr_i81++) {
            Float __cfwr_data74 = null;
        }
ative int i0 = s.length() - 10;
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

    public Boolean __cfwr_util113(boolean __cfwr_p0, Double __cfwr_p1) {
        for (int __cfwr_i27 = 0; __cfwr_i27 < 1; __cfwr_i27++) {
            return null;
        }
        return (-53L & -58.73);
        return null;
    }
    private Object __cfwr_process636(Double __cfwr_p0, Long __cfwr_p1) {
        while (true) {
            for (int __cfwr_i43 = 0; __cfwr_i43 < 5; __cfwr_i43++) {
            for (int __cfwr_i96 = 0; __cfwr_i96 < 4; __cfwr_i96++) {
            for (int __cfwr_i91 = 0; __cfwr_i91 < 5; __cfwr_i91++) {
            Object __cfwr_entry16 = null;
        }
        }
        }
            break; // Prevent infinite loops
        }
        try {
            try {
            if (true || (false ^ -47.24f)) {
            try {
            return ((-68.46f ^ false) + (-20L * null));
        } catch (Exception __cfwr_e67) {
            // ignore
        }
        }
        } catch (Exception __cfwr_e45) {
            // ignore
        }
        } catch (Exception __cfwr_e25) {
            // ignore
        }
        for (int __cfwr_i81 = 0; __cfwr_i81 < 7; __cfwr_i81++) {
            for (int __cfwr_i44 = 0; __cfwr_i44 < 8; __cfwr_i44++) {
            try {
            if (true && (false + true)) {
            if (true && true) {
            return null;
        }
        }
        } catch (Exception __cfwr_e15) {
            // ignore
        }
        }
        }
        return null;
    }
}