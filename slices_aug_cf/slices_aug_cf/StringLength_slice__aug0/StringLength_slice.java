/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void testNewArraySameLen(String s) {
        for (int __cfwr_i66 = 0; __cfwr_i66 < 10; __cfwr_i66++) {
            while (false) {
            try {
            if (false || ((null | 32.50) | (null * null))) {
            try {
            try {
            if (true || false) {
            double __cfwr_result2 = ((null + true) * -492);
        }
        } catch (Exception __cfwr_e13) {
            // ignore
        }
        } catch (Exception __cfwr_e86) {
            // ignore
        }
        }
        } catch (Exception __cfwr_e26) {
            // ignore
        }
            break; // Prevent infinite loops
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
      protected static Long __cfwr_compute289(String __cfwr_p0, double __cfwr_p1, Long __cfwr_p2) {
        for (int __cfwr_i70 = 0; __cfwr_i70 < 1; __cfwr_i70++) {
            try {
            while (true) {
            while (false) {
            try {
            if (false && true) {
            if (false || (true % '3')) {
            long __cfwr_temp27 = (false | 829);
        }
        }
        } catch (Exception __cfwr_e91) {
            // ignore
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e50) {
            // ignore
        }
        }
        for (int __cfwr_i66 = 0; __cfwr_i66 < 10; __cfwr_i66++) {
            Object __cfwr_entry58 = null;
        }
        for (int __cfwr_i88 = 0; __cfwr_i88 < 2; __cfwr_i88++) {
            try {
            float __cfwr_elem36 = 27.65f;
        } catch (Exception __cfwr_e29) {
            // ignore
        }
        }
        try {
            if (false || true) {
            try {
            for (int __cfwr_i62 = 0; __cfwr_i62 < 1; __cfwr_i62++) {
            while (false) {
            return ('y' * 16.54);
            break; // Prevent infinite loops
        }
        }
        } catch (Exception __cfwr_e99) {
            // ignore
        }
        }
        } catch (Exception __cfwr_e42) {
            // ignore
        }
        return null;
    }
    protected static long __cfwr_compute792(Object __cfwr_p0, short __cfwr_p1) {
        if (true && false) {
            double __cfwr_node36 = (('8' / null) << false);
        }
        return 676L;
    }
}
