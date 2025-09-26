/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void testNewArraySameLen(String s) {
        for (int __cfwr_i13 = 0; __cfwr_i13 < 4; __cfwr_i13++) {
            for (int __cfwr_i31 = 0; __cfwr_i31 < 7; __cfwr_i31++) {
            for (int __cfwr_i59 = 0; __cfwr_i59 < 10; __cfwr_i59++)
        return null;
 {
            if (false && true) {
            while ((32.32f - (194 | 84.69f))) {
            return "world30";
            break; // Prevent infinite loops
        }
        }
        }
        }
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
      Object __cfwr_func446(Boolean __cfwr_p0, Boolean __cfwr_p1) {
        for (int __cfwr_i81 = 0; __cfwr_i81 < 10; __cfwr_i81++) {
            if (true && false) {
            return "world45";
        }
        }
        try {
            for (int __cfwr_i65 = 0; __cfwr_i65 < 8; __cfwr_i65++) {
            char __cfwr_entry24 = '0';
        }
        } catch (Exception __cfwr_e64) {
            // ignore
        }
        Float __cfwr_elem29 = null;
        for (int __cfwr_i82 = 0; __cfwr_i82 < 5; __cfwr_i82++) {
            if (false || true) {
            try {
            while (false) {
            if ((954 - ('7' - 458L)) && true) {
            if (((467 + 'e') << null) || true) {
            return null;
        }
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e91) {
            // ignore
        }
        }
        }
        return null;
    }
    private String __cfwr_helper287() {
        while (true) {
            try {
            if ((-60.97 / 'M') || true) {
            Character __cfwr_temp57 = null;
        }
        } catch (Exception __cfwr_e10) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        while ((false % 89.86)) {
            if (true && (false >> null)) {
            return 'O';
        }
            break; // Prevent infinite loops
        }
        if (true && false) {
            return (-108L & null);
        }
        return "world30";
    }
}
