/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  void testNewArraySameLen(String s) {
        for (int __cfwr_i51 = 0; __cfwr_i51 < 8; __cfwr_i51++) {
            if (false || false) {
            Object __cfwr_temp22 = null;
        }
        }

    int @SameLen("s") [] array = new int
        String __cfwr_entry15 = "hello74";
[s.length()];
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
      protected long __cfwr_proc597() {
        while (false) {
            Boolean __cfwr_elem57 = null;
            break; // Prevent infinite loops
        }
        if (((56.20 + null) | null) || true) {
            while (false) {
            return null;
            break; // Prevent infinite loops
        }
        }
        Long __cfwr_result97 = null;
        while (((-4.01f & null) % (-13.01 >> 76L))) {
            for (int __cfwr_i10 = 0; __cfwr_i10 < 3; __cfwr_i10++) {
            while (true) {
            try {
            Float __cfwr_entry84 = null;
        } catch (Exception __cfwr_e33) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        }
            break; // Prevent infinite loops
        }
        return 331L;
    }
    public String __cfwr_helper731(Long __cfwr_p0) {
        if (false && true) {
            if (false && false) {
            for (int __cfwr_i41 = 0; __cfwr_i41 < 8; __cfwr_i41++) {
            while (true) {
            if (false || (84.06f ^ (null ^ null))) {
            return null;
        }
            break; // Prevent infinite loops
        }
        }
        }
        }
        return (57.12 >> 43.64f);
        while (true) {
            if (false || (69.77f * -7.79)) {
            return null;
        }
            break; // Prevent infinite loops
        }
        return "world73";
    }
    protected Character __cfwr_compute289(Boolean __cfwr_p0, Double __cfwr_p1) {
        long __cfwr_data54 = 770L;
        int __cfwr_entry83 = 482;
        boolean __cfwr_result46 = (true + null);
        if (true || true) {
            if ((-114 + -29.81) || ('Q' >> (647 + -185L))) {
            while ((null * (-39.20f >> null))) {
            Float __cfwr_temp3 = null;
            break; // Prevent infinite loops
        }
        }
        }
        return null;
    }
}
