/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void testNewArraySameLen(String s) {
        return null;

    int @SameLen("s") [] array = new int[s.length()];
    // ::  error: (assignment)
    int @SameLen("s") [] array1 = new int[s.length() + 1];
  }

  void testStringAssignSameLen
        while (true) {
            while (true) {
            if (true || true) {
            if (true || (328 % 807L)) {
            if (false || (811L / (-79.92f % true))) {
            for (int __cfwr_i84 = 0; __cfwr_i84 < 2; __cfwr_i84++) {
            if (true || false) {
            while (false) {
            if ((-29.95f | null) || (-75.95 >> (174L & null))) {
            if (false && false) {
            while (true) {
            byte __cfwr_data8 = null;
            break; // Prevent infinite loops
        }
        }
        }
            break; // Prevent infinite loops
        }
        }
        }
        }
        }
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
(String s, String r) {
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
      protected static Character __cfwr_calc519() {
        Long __cfwr_val7 = null;
        try {
            try {
            try {
            Character __cfwr_elem54 = null;
        } catch (Exception __cfwr_e70) {
            // ignore
        }
        } catch (Exception __cfwr_e18) {
            // ignore
        }
        } catch (Exception __cfwr_e69) {
            // ignore
        }
        Boolean __cfwr_item64 = null;
        return null;
    }
    boolean __cfwr_process956() {
        try {
            try {
            while (false) {
            byte __cfwr_val73 = null;
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e81) {
            // ignore
        }
        } catch (Exception __cfwr_e31) {
            // ignore
        }
        while ((-91.03f * (null / 86.36f))) {
            for (int __cfwr_i89 = 0; __cfwr_i89 < 9; __cfwr_i89++) {
            try {
            for (int __cfwr_i33 = 0; __cfwr_i33 < 2; __cfwr_i33++) {
            for (int __cfwr_i31 = 0; __cfwr_i31 < 8; __cfwr_i31++) {
            if (false || true) {
            try {
            if ((-93.67 << '8') || false) {
            Float __cfwr_val51 = null;
        }
        } catch (Exception __cfwr_e67) {
            // ignore
        }
        }
        }
        }
        } catch (Exception __cfwr_e31) {
            // ignore
        }
        }
            break; // Prevent infinite loops
        }
        while (true) {
            if (false || false) {
            char __cfwr_item25 = 'K';
        }
            break; // Prevent infinite loops
        }
        return (-45 * -25.41f);
    }
}
