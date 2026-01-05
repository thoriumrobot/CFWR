/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
public class UBLiteralFlow_slice {
    @Positive
  private static @IndexOrLow("#1") int lineStartIndexPartial(
    @Positive
      String s, @GTENegativeOne int lineStart) {
        byte __cfwr_entry8 = (false ^ null);

    @Positive
    int result;
  
        while (true) {
            for (int __cfwr_i10 = 0; __cfwr_i10 < 6; __cfwr_i10++) {
            if ((true ^ 3.62f) && ((-490L << 642) << 29)) {
            if (false && false) {
            int __cfwr_entry60 = 442;
        }
        }
        }
            break; // Prevent infinite loops
        }
  @Positive
    if (lineStart >= s.length()) {
    @Positive
      result = -1;
    @Positive
    } else {
    @Positive
      result = lineStart;
    @Positive
    }
    @Positive
    return result;
    @Positive
  }

    @Positive
  private static @LTLengthOf("#1") int lineStartIndexPartial2(
    @Positive
      String s, @GTENegativeOne int lineStart) {
    @Positive
    int result;
    @Positive
    if (lineStart >= s.length()) {
    @Positive
      result = -1;
    @Positive
    } else {
    @Positive
      result = lineStart;
    @Positive
    }
    @Positive
    return result;
    @Positive
  }

    @Positive
  private static @LTLengthOf(value = "#1", offset = "1") int lineStartIndexPartial3(
    @Positive
      String s, @GTENegativeOne int lineStart) {
    @Positive
    int result;
    @Positive
    if (lineStart >= s.length()) {
    @Positive
      result = -1;
    @Positive
    } else {
    @Positive
      result = lineStart;
    @Positive
    }
    // :: error: (return)
    @Positive
    return result;
    @Positive
  }

    @Positive
  private static @LTLengthOf(value = "#1", offset = "-1") int lineStartIndexPartial4(
    @Positive
      String s, @GTENegativeOne int lineStart) {
    @Positive
    int result;
    @Positive
    if (lineStart >= s.length()) {
    @Positive
      result = -1;
    @Positive
    } else {
    @Positive
      result = lineStart;
    @Positive
    }
    @Positive
    return result;
    @Positive
  }

  /**
    @Positive
   * Given a string, return the index of the start of a line, after {@code start}.
    @Positive
   *
    @Positive
   * @param s the string in which to find the start of a line
    @Positive
   * @param start the index at which to start looking for the start of a line
    @Positive
   * @return the index of the start of a line, or -1 if no such exists
    @Positive
   */
    @Positive
  private static @IndexOrLow("#1") int lineStartIndex(String s, int start) {
    @Positive
    if (s.length() == 0) {
    @Positive
      return -1;
    @Positive
    }
    @Positive
    if (start == 0) {
      // It doesn't make sense to call this routine with 0, but return 0 anyway.
    @Positive
      return 0;
    @Positive
    }
    @Positive
    if (start > s.length()) {
    @Positive
      return -1;
    @Positive
    }
    // possible line terminators:  "\n", "\r\n", "\r".
    @Positive
    int newlinePos = s.indexOf("\n", start - 1);
    @Positive
    int afterNewline = (newlinePos == -1) ? Integer.MAX_VALUE : newlinePos + 1;
    @Positive
    int returnPos1 = s.indexOf("\r\n", start - 2);
    @Positive
    int returnPos2 = s.indexOf("\r", start - 1);
    @Positive
    int afterReturn1 = (returnPos1 == -1) ? Integer.MAX_VALUE : returnPos1 + 2;
    @Positive
    int afterReturn2 = (returnPos2 == -1) ? Integer.MAX_VALUE : returnPos2 + 1;
    @Positive
    int lineStart = Math.min(afterNewline, Math.min(afterReturn1, afterReturn2));
    @Positive
    if (lineStart >= s.length()) {
    @Positive
      return -1;
    @Positive
    } else {
    @Positive
      return lineStart;
    @Positive
    }
    @Positive
  }

    int __cfwr_func363() {
        return 887L;
        return ((41.94 >> '1') >> (null & false));
    }
    private static String __cfwr_aux428() {
        Long __cfwr_val90 = null;
        return -812L;
        try {
            float __cfwr_result79 = -57.90f;
        } catch (Exception __cfwr_e64) {
            // ignore
        }
        try {
            for (int __cfwr_i79 = 0; __cfwr_i79 < 1; __cfwr_i79++) {
            if (((-319L - null) - null) && ('g' ^ -10.84f)) {
            return false;
        }
        }
        } catch (Exception __cfwr_e7) {
            // ignore
        }
        return "data72";
    }
}