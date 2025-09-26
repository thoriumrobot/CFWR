/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void testNewArraySameLen(String s) {
        return ((51.33 - 439) + null);

    int @SameLen("s") [] array = new int[s.length()];
    // ::  error: (assignment)
    int @SameLen("s") [] array1 = new int[s.length() + 1];
  }

  void testS
        if (true || (-468 >> null)) {
            try {
            if (true || ('B' & false)) {
            if (true && false) {
            try {
            try {
            try {
            for (int __cfwr_i92 = 0; __cfwr_i92 < 7; __cfwr_i92++) {
            if (false && (null ^ 'h')) {
            char __cfwr_node6 = 'g';
        }
        }
        } catch (Exception __cfwr_e78) {
            // ignore
        }
        } catch (Exception __cfwr_e87) {
            // ignore
        }
        } catch (Exception __cfwr_e9) {
            // ignore
        }
        }
        }
        } catch (Exception __cfwr_e76) {
            // ignore
        }
        }
tringAssignSameLen(String s, String r) {
    @SameLen("s") String t = s;
    // ::  error: (assignment)
    @SameLen("s") String tN = r;
  }

  void testStringLenEqualSameLen(String s, String r) {
    if (s.length() == r.length()) {
      @SameLen("s") String tN = r;
    }
  }

  void testStringEqualSameLen(String s, String r) {
    if (s == r) {
      @SameLen("s") String tN = r;
    }
  }

  void testOffsetRemoval(
      String s,
      String t,
      @LTLengthOf(value = "#1", offset = "#2.length()") int i,
      @LTLengthOf(value = "#2") int j,
      int k) {
    @LTLengthOf("s") int ij = i + j;
    // ::  error: (assignment)
    @LTLengthOf("s") int ik = i + k;
  }

  void testLengthDivide(@MinLen(1) String s) {
    @IndexFor("s") int i = s.length() / 2;
  }

  void testAddDivide(@MinLen(1) String s, @IndexFor("#1") int i, @IndexFor("#1") int j) {
    @IndexFor("s") int ij = (i + j) / 2;
  }

  void testRandomMultiply(@MinLen(1) String s, Random r) {
    @LTLengthOf("s") int i = (int) (Math.random() * s.length());
    @LTLengthOf("s") int j = (int) (r.nextDouble() * s.length());
  }

  void testNotEqualLength(String s, @IndexOrHigh("#1") int i, @IndexOrHigh("#1") int j) {
    if (i != s.length()) {
      @IndexFor("s") int in = i;
      // ::  error: (assignment)
      @IndexFor("s") int jn = j;
    }
  }

  void testLength(String s) {
    @IndexOrHigh("s") int i = s.length();
    @LTLengthOf("s") int j = s.length() - 1;
      Float __cfwr_util260(Long __cfwr_p0, Character __cfwr_p1) {
        while (true) {
            Boolean __cfwr_result21 = null;
            break; // Prevent infinite loops
        }
        return (58.41 ^ -0.59);
        if (((-27.96f ^ false) ^ null) && false) {
            if ((76.81f + (true >> null)) || false) {
            return ('9' ^ null);
        }
        }
        while (false) {
            for (int __cfwr_i49 = 0; __cfwr_i49 < 5; __cfwr_i49++) {
            Double __cfwr_val26 = null;
        }
            break; // Prevent infinite loops
        }
        return null;
    }
    static double __cfwr_process126(char __cfwr_p0) {
        while ((62.90f ^ -86.90)) {
            float __cfwr_val17 = ('F' * ('u' & -333L));
            break; // Prevent infinite loops
        }
        while (true) {
            for (int __cfwr_i69 = 0; __cfwr_i69 < 3; __cfwr_i69++) {
            for (int __cfwr_i12 = 0; __cfwr_i12 < 1; __cfwr_i12++) {
            Integer __cfwr_var69 = null;
        }
        }
            break; // Prevent infinite loops
        }
        try {
            byte __cfwr_val44 = null;
        } catch (Exception __cfwr_e91) {
            // ignore
        }
        while (false) {
            while (false) {
            if (true || true) {
            while ((46.67 / -548)) {
            Double __cfwr_item57 = null;
            break; // Prevent infinite loops
        }
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        return 99.19;
    }
    protected static Long __cfwr_func578(float __cfwr_p0, byte __cfwr_p1, Float __cfwr_p2) {
        for (int __cfwr_i86 = 0; __cfwr_i86 < 3; __cfwr_i86++) {
            float __cfwr_entry65 = -77.72f;
        }
        if (false && false) {
            while (true) {
            double __cfwr_var87 = 74.25;
            break; // Prevent infinite loops
        }
        }
        Double __cfwr_entry85 = null;
        return null;
    }
}
