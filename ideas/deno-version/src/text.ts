export function isLetter(b: number): boolean {
  return (b >= 65 && b <= 90) || (b >= 97 && b <= 122);
}

export const encoder = new TextEncoder();
export const decoder = new TextDecoder("utf-8", { fatal: false });

export function renderBytes(bytes: ArrayLike<number>): string {
  let s = "";
  for (let i = 0; i < bytes.length; i++) {
    const b = bytes[i];
    if (b === 10) s += "\\n";
    else if (b === 32) s += "␣";
    else if (b >= 33 && b < 127) s += String.fromCharCode(b);
    else s += `\\x${b.toString(16).padStart(2, "0")}`;
  }
  return s;
}
