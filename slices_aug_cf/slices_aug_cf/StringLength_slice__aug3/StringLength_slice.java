/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void testNewArraySameLen(String s) {
        while ((76.41 & null)) {
            try {
            Long __cfwr_temp52 = null;
        } catch (Exception __cfwr_e98) {
            // ignore
        }
            break; // Prevent infinite loops
        }

    int @SameLen("s") [] array = new int[s.length()];
    // ::  error: (assignment)
    int @SameLen("s") [] array1 = new int[s.length() + 1];
  }

  void testStringAssignSameLen(String s, String r) {
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
      protected static Object __cfwr_aux851(int __cfwr_p0, double __cfwr_p1, Character __cfwr_p2) {
        try {
            if (false && ((null - '0') % 'z')) {
            try {
            for (int __cfwr_i36 = 0; __cfwr_i36 < 7; __cfwr_i36++) {
            Character __cfwr_var84 = null;
        }
        } catch (Exception __cfwr_e19) {
            // ignore
        }
        }
        } catch (Exception __cfwr_e26) {
            // ignore
        }
        return null;
    }
    protected Boolean __cfwr_calc121() {
        if (true || (null % true)) {
            if (true || true) {
            while (true) {
            if (false || (null * -84L)) {
            float __cfwr_var84 = (-36.52f % (-58.89 * null));
        }
            break; // Prevent infinite loops
        }
        }
        }
        return null;
    }
}
