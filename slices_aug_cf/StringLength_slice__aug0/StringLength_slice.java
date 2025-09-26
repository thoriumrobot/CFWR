/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class StringLength_slice {
  void testNewArraySameLen(String s) {
        for (int __cfwr_i3 = 0; __cfwr_i3 < 8; __cfwr_i3++) {
            for (int __cfwr_i35 = 0; __cfwr_i35 < 1; __cfwr_i35++) {
            for (int __cfwr_i15 = 0; __cfwr_i15 < 5; __cfwr_i15++) {
            return ('e' >> -166);
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
  }

    public static Character __cfwr_helper301() {
        return null;
        if ((-832 << true) || (null >> (false / 'q'))) {
            while (true) {
            for (int __cfwr_i75 = 0; __cfwr_i75 < 5; __cfwr_i75++) {
            if (false || false) {
            return "temp45";
        }
        }
            break; // Prevent infinite loops
        }
        }
        return null;
        return null;
    }
    static byte __cfwr_compute409(float __cfwr_p0) {
        try {
            if (false || (514 << -32.26f)) {
            try {
            try {
            for (int __cfwr_i95 = 0; __cfwr_i95 < 2; __cfwr_i95++) {
            while (((null * 860L) - (17 - null))) {
            while ((827 ^ (null + null))) {
            while (((null ^ 37.90f) << null)) {
            return 46.22f;
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        }
        } catch (Exception __cfwr_e47) {
            // ignore
        }
        } catch (Exception __cfwr_e50) {
            // ignore
        }
        }
        } catch (Exception __cfwr_e52) {
            // ignore
        }
        while (true) {
            while (false) {
            return (null ^ 'c');
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        long __cfwr_obj91 = (-89.03 / 'a');
        return null;
        return null;
    }
}