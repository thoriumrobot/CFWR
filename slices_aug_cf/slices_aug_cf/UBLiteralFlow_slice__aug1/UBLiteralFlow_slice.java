/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
  private static @IndexOrLow("#1") int lineStartIndexPartial(
      String s, @GTENegativeOne int lineStart) {
        for (int __cfwr_i40 = 0; __cfwr_i40 < 2; __cfwr_i40++) {
            while (false) {
            return null;
            break; // Prevent infinite loops
        }
        }

    int result;
    if (lineStart >= s.length()) {
      result = -1;
    } else {
      result = lineStart;
    }
    return result;
  }

  private static @LTLengthOf("#1") int lineStartIndexPartial2(
      String s, @GTENegativeOne int lineStart) {
    int result;
    if (lineStart >= s.length()) {
      result = -1;
    } else {
      result = lineStart;
    }
    return result;
  }

  private static @LTLengthOf(value = "#1", offset = "1") int lineStartIndexPartial3(
      String s, @GTENegativeOne int lineStart) {
    int result;
    if (lineStart >= s.length()) {
      result = -1;
    } else {
      result = lineStart;
    }
    // :: error: (return)
    return result;
  }

  private static @LTLengthOf(value = "#1", offset = "-1") int lineStartIndexPartial4(
      String s, @GTENegativeOne int lineStart) {
    int result;
    if (lineStart >= s.length()) {
      result = -1;
    } else {
      result = lineStart;
    }
    return result;
  }

  /**
   * Given a string, return the index of the start of a line, after {@code start}.
   *
   * @param s the string in which to find the start of a line
   * @param start the index at which to start looking for the start of a line
   * @return the index of the start of a line, or -1 if no such exists
   */
  private static @IndexOrLow("#1") int lineStartIndex(String s, int start) {
    if (s.length() == 0) {
      return -1;
    }
    if (start == 0) {
      // It doesn't make sense to call this routine with 0, but return 0 anyway.
      return 0;
    }
    if (start > s.length()) {
      return -1;
    }
    // possible line terminators:  "\n", "\r\n", "\r".
    int newlinePos = s.indexOf("\n", start - 1);
    int afterNewline = (newlinePos == -1) ? Integer.MAX_VALUE : newlinePos + 1;
    int returnPos1 = s.indexOf("\r\n", start - 2);
    int returnPos2 = s.indexOf("\r", start - 1);
    int afterReturn1 = (returnPos1 == -1) ? Integer.MAX_VALUE : returnPos1 + 2;
    int afterReturn2 = (returnPos2 == -1) ? Integer.MAX_VALUE : returnPos2 + 1;
    int lineStart = Math.min(afterNewline, Math.min(afterReturn1, afterReturn2));
    if (lineStart >= s.length()) {
      return -1;
    } else {
      return lineStart;
    }
  }

  /**
   * Given a string, return the index of the start of a line, after {@code start}.
   *
   * @param s the string in which to find the start of a line
   * @param start the index at which to start looking for the start of a line
   * @return the index of the start of a line, or -1 if no such exists
   */
  private static @LTLengthOf("#1") int lineStartIndex2(String s, int start) {
    if (s.length() == 0) {
      return -1;
    }
    if (start == 0) {
      // It doesn't make sense to call this routine with 0, but return 0 anyway.
      return 0;
    }
    if (start > s.length()) {
      return -1;
    }
    // possible line terminators:  "\n", "\r\n", "\r".
    int newlinePos = s.indexOf("\n", start - 1);
    int afterNewline = (newlinePos == -1) ? Integer.MAX_VALUE : newlinePos + 1;
    int returnPos1 = s.indexOf("\r\n", start - 2);
    int returnPos2 = s.indexOf("\r", start - 1);
    int afterReturn1 = (returnPos1 == -1) ? Integer.MAX_VALUE : returnPos1 + 2;
    int afterReturn2 = (returnPos2 == -1) ? Integer.MAX_VALUE : returnPos2 + 1;
    int lineStart = Math.min(afterNewline, Math.min(afterReturn1, afterReturn2));
    if (lineStart >= s.length()) {
      return -1;
    } else {
      return lineStart;
    }
  }

  /**
   * Given a string, return the index of the start of a line, after {@code start}.
   *
   * @param s the string in which to find the start of a line
   * @param start the index at which to start looking for the start of a line
   * @return the index of the start of a line, or -1 if no such exists
   */
  private static @LTLengthOf(value = "#1", offset = "1") int lineStartIndex3(String s, int start) {
    if (s.length() == 0) {
      // :: error: (return)
      return -1;
    }
    if (start == 0) {
      // It doesn't make sense to call this routine with 0, but return 0 anyway.
      // :: error: (return)
      return 0;
    }
    if (start > s.length()) {
      return -1;
    }
    // possible line terminators:  "\n", "\r\n", "\r".
    int newlinePos = s.indexOf("\n", start - 1);
    int afterNewline = (newlinePos == -1) ? Integer.MAX_VALUE : newlinePos + 1;
    int returnPos1 = s.indexOf("\r\n", start - 2);
    int returnPos2 = s.indexOf("\r", start - 1);
    int afterReturn1 = (returnPos1 == -1) ? Integer.MAX_VALUE : returnPos1 + 2;
    int afterReturn2 = (returnPos2 == -1) ? Integer.MAX_VALUE : returnPos2 + 1;
    int lineStart = Math.min(afterNewline, Math.min(afterReturn1, afterReturn2));
    if (lineStart >= s.length()) {
      return -1;
    } else {
      // :: error: (return)
      return lineStart;
    }
  }

  /**
   * Given a string, return the index of the start of a line, after {@code start}.
   *
   * @param s the string in which to find the start of a line
   * @param start the index at which to start looking for the start of a line
   * @return the index of the start of a line, or -1 if no such exists
   */
  private static @LTLengthOf(value = "#1", offset = "-1") int lineStartIndex4(String s, int start) {
    if (s.length() == 0) {
      return -1;
    }
    if (start == 0) {
      // It doesn't make sense to call this routine with 0, but return 0 anyway.
      return 0;
    }
    if (start > s.length()) {
      return -1;
    }
    // possible line terminators:  "\n", "\r\n", "\r".
    int newlinePos = s.indexOf("\n", start - 1);
    int afterNewline = (newlinePos == -1) ? Integer.MAX_VALUE : newlinePos + 1;
    int returnPos1 = s.indexOf("\r\n", start - 2);
    int returnPos2 = s.indexOf("\r", start - 1);
    int afterReturn1 = (returnPos1 == -1) ? Integer.MAX_VALUE : returnPos1 + 2;
    int afterReturn2 = (returnPos2 == -1) ? Integer.MAX_VALUE : returnPos2 + 1;
    int lineStart = Math.min(afterNewline, Math.min(afterReturn1, afterReturn2));
    if (lineStart >= s.length()) {
      return -1;
    } else {
      return lineStart;
    }
      protected static String __cfwr_temp680(float __cfwr_p0) {
        try {
            while (false) {
            try {
            short __cfwr_data84 = null;
        } catch (Exception __cfwr_e9) {
            // ignore
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e47) {
            // ignore
        }
        double __cfwr_obj49 = 60.67;
        String __cfwr_elem77 = "result94";
        return "value8";
    }
    protected static Integer __cfwr_util620(String __cfwr_p0) {
        while (true) {
            return 37.14;
            break; // Prevent infinite loops
        }
        while (true) {
            Float __cfwr_val82 = null;
            break; // Prevent infinite loops
        }
        return null;
    }
}
